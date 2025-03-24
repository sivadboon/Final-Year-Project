import os
import torch
import json
import openai
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set environment variables (for HF code execution)
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Authenticate Hugging Face
login(HUGGINGFACE_API_TOKEN)

# Load HumanEval dataset and evaluation metric
human_eval = load_dataset("openai_humaneval")["test"]
code_eval_metric = load("code_eval")

# Define models and settings
MODELS = [
    "bigcode/starcoderbase-7b",       # Already evaluated (will be skipped)    
    "gpt-4-api",                      # OpenAI GPT-4
    "meta-llama/CodeLlama-7b-Python-hf",     # CodeLlama
]

num_samples_per_problem = 5
k_values = [1, 5]

# Create results directory
os.makedirs("evaluation/results", exist_ok=True)

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# GPT-4 generation helper
def generate_gpt4_code(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.7,
    )
    return response.choices[0].message.content

# Evaluation pipeline
def evaluate_model(model_name):
    results_path = f"evaluation/results/{model_name.replace('/', '_')}_benchmark_results.json"
    if os.path.exists(results_path):
        print(f"⏭️ Skipping {model_name} — already evaluated.")
        return

    print(f"🚀 Evaluating Model: {model_name}")
    test_cases = []
    candidates = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "gpt-4-api":
        print("🔹 Using OpenAI GPT-4 API for evaluation")
        for problem in tqdm(human_eval, desc="Problems", unit="problem"):
            prompt = problem["prompt"]
            test_cases.append(problem["test"])
            problem_candidates = [
                generate_gpt4_code(prompt) for _ in range(num_samples_per_problem)
            ]
            candidates.append(problem_candidates)
    else:
        # Load Hugging Face model with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HUGGINGFACE_API_TOKEN,
            quantization_config=bnb_config
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGINGFACE_API_TOKEN)
        model.eval()

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0

        for problem in tqdm(human_eval, desc="Problems", unit="problem"):
            prompt = problem["prompt"]
            test_cases.append(problem["test"])
            problem_candidates = []

            for _ in range(num_samples_per_problem):
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_code = generated_code[len(prompt):]
                problem_candidates.append(generated_code)

            candidates.append(problem_candidates)

    # Save generations
    generations_path = f"evaluation/results/{model_name.replace('/', '_')}_generations.json"
    with open(generations_path, "w") as f:
        json.dump({"references": test_cases, "predictions": candidates}, f)
    print(f"Code generation complete. Saved at {generations_path}")

    # Evaluate pass@k
    print(f"🔍 Evaluating pass@k for {model_name}...")
    pass_at_k, _ = code_eval_metric.compute(
        references=test_cases,
        predictions=candidates,
        k=k_values,
        num_workers=4,
        timeout=10.0,
    )

    with open(results_path, "w") as f:
        json.dump(pass_at_k, f)

    print(f"Evaluation complete for {model_name}. Results saved at {results_path}")
    for k in k_values:
        print(f"🎯 Pass@{k}: {pass_at_k[f'pass@{k}'] * 100:.2f}%")

# Run evaluations
for model in MODELS:
    evaluate_model(model)

print("All model evaluations complete! Check `evaluation/results/` for outputs.")
