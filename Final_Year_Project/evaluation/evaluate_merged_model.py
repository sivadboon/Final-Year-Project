import os
import torch
import json
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

# Config
MERGED_MODEL_PATH = "merged_model"  # Path to your merged model
NUM_SAMPLES_PER_PROBLEM = 5
K_VALUES = [1, 5]
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load HumanEval dataset and evaluation metric
human_eval = load_dataset("openai_humaneval")["test"]
code_eval_metric = load("code_eval")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
print(f"🚀 Loading model from {MERGED_MODEL_PATH}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
).eval()

tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0

# Generate and evaluate
test_cases = []
candidates = []

print("🧪 Generating samples...")
for problem in tqdm(human_eval, desc="Problems"):
    prompt = problem["prompt"]
    test_cases.append(problem["test"])
    problem_candidates = []

    for _ in range(NUM_SAMPLES_PER_PROBLEM):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_code = generated_code[len(prompt):]
        problem_candidates.append(generated_code)

    candidates.append(problem_candidates)

# Save generations
os.makedirs("evaluation/results", exist_ok=True)
gen_path = "evaluation/results/merged_model_generations.json"
with open(gen_path, "w") as f:
    json.dump({"references": test_cases, "predictions": candidates}, f)
print(f"💾 Generated samples saved at {gen_path}")

# Compute pass@k
print("📊 Evaluating pass@k...")
pass_at_k, _ = code_eval_metric.compute(
    references=test_cases,
    predictions=candidates,
    k=K_VALUES,
    num_workers=4,
    timeout=10.0,
)

# Save evaluation results
results_path = "evaluation/results/merged_model_eval_results.json"
with open(results_path, "w") as f:
    json.dump(pass_at_k, f)

print(f"Evaluation complete. Results saved at {results_path}")
for k in K_VALUES:
    print(f"🎯 Pass@{k}: {pass_at_k[f'pass@{k}'] * 100:.2f}%")
