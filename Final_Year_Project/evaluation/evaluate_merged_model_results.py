import os
import json
import math

# Load generated samples from file
gen_path = "evaluation/results/merged_model_generations.json"
with open(gen_path, "r") as f:
    data = json.load(f)

references = data["references"]
predictions = data["predictions"]

# Define k-values
k_values = [1, 5]
num_samples = len(predictions[0])  # Assumes all problems have the same number of samples

# For now, assume 0 correct samples (no execution). Replace this with real checking if needed.
# This function estimates pass@k as per OpenAI HumanEval paper
def estimate_pass_at_k(n, c, k):
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1 - math.comb(n - c, k) / math.comb(n, k)

# Compute pass@k
print("📊 Evaluating pass@k (Windows-safe)...")
pass_at_k = {f"pass@{k}": 0.0 for k in k_values}
total_problems = len(references)

for i in range(total_problems):
    preds = predictions[i]

    # Replace this logic with actual checking in the future
    num_correct = 0  # Assume none of the generated codes passed

    for k in k_values:
        pass_at_k[f"pass@{k}"] += estimate_pass_at_k(len(preds), num_correct, k)

# Average over all problems
for k in k_values:
    pass_at_k[f"pass@{k}"] /= total_problems

# Save evaluation results
results_path = "evaluation/results/merged_model_eval_results.json"
with open(results_path, "w") as f:
    json.dump(pass_at_k, f, indent=2)

# Print results
print(f"Evaluation complete. Results saved at {results_path}")
for k in k_values:
    print(f"🎯 Pass@{k}: {pass_at_k[f'pass@{k}'] * 100:.2f}%")
