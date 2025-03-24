import os
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from evaluate import load
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set environment variables (for HF code execution)
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Authenticate Hugging Face
login(HUGGINGFACE_API_TOKEN)

# Configs
MODEL_NAME = "meta-llama/CodeLlama-7b-Python-hf"  
NUM_SAMPLES_PER_PROBLEM = 5
K_VALUES = [1, 5]
RESULTS_DIR = "evaluation/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load dataset and metric
human_eval = load_dataset("openai_humaneval")["test"]
code_eval = load("code_eval")

# Load model and tokenizer (4-bit for memory efficiency)
print(f"🚀 Loading model: {MODEL_NAME}")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    token=HUGGINGFACE_API_TOKEN
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HUGGINGFACE_API_TOKEN
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0

# Generation loop
print("🧪 Generating completions...")
references = []
predictions = []

for problem in tqdm(human_eval, desc="Problems", unit="problem"):
    prompt = problem["prompt"]
    references.append(problem["test"])
    problem_candidates = []

    for _ in range(NUM_SAMPLES_PER_PROBLEM):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=1,
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Try removing the prompt from the start of the output (if duplicated)
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()

        problem_candidates.append(generated)

    predictions.append(problem_candidates)

# Save generations
gen_file = os.path.join(RESULTS_DIR, "codellama_7b_python_generations.json")
with open(gen_file, "w") as f:
    json.dump({"references": references, "predictions": predictions}, f)
print(f"💾 Generations saved to {gen_file}")

# Evaluate pass@k
print("📊 Computing pass@k...")
pass_at_k, _ = code_eval.compute(
    references=references,
    predictions=predictions,
    k=K_VALUES,
    num_workers=4,
    timeout=10.0,
)

# Save results
result_file = os.path.join(RESULTS_DIR, "codellama_7b_python_eval_results.json")
with open(result_file, "w") as f:
    json.dump(pass_at_k, f, indent=2)

# Print summary
print("Evaluation complete!")
for k in K_VALUES:
    print(f"🎯 Pass@{k}: {pass_at_k[f'pass@{k}'] * 100:.2f}%")