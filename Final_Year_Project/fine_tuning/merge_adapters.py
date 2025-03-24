import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
import os

# Configuration
base_model = "bigcode/starcoderbase-7b"
peft_model_path = "fine_tuned_model"
offload_dir = "offload"
os.makedirs(offload_dir, exist_ok=True)
output_path = "merged_model"

print("Merging adapter...")

# Load base model with offloading
base_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    offload_folder=offload_dir,
    torch_dtype=torch.float16,
)

base_model.config.use_cache = False

# Load PEFT adapter with offloading
model = PeftModel.from_pretrained(
    base_model,
    peft_model_path,
    offload_folder=offload_dir
)

# Merge and save
model = model.merge_and_unload()
model.save_pretrained(output_path)
print(f"Merged model saved at {output_path}")
