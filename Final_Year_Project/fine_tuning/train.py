import os
import torch
from dotenv import load_dotenv  
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
import wandb
import matplotlib.pyplot as plt

# Load API keys from .env
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Free CUDA memory
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# Login
login(token=HUGGINGFACE_TOKEN)
wandb.login(key=WANDB_API_KEY)
wandb.login(key=WANDB_API_KEY)

# Config
MODEL_NAME = "bigcode/starcoderbase-7b"
PROCESSED_DATASET = "fine_tuning/processed_dataset"
OUTPUT_DIR = "fine_tuned_model"

# Load dataset
print("Loading dataset...")
dataset = load_from_disk(PROCESSED_DATASET)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Tokenization
print("🔍 Tokenizing...")
def tokenize(sample):
    input_text = f"<|user|>\n{sample['question']}\n<|assistant|>\n{sample['response']}"
    tokenized = tokenizer(input_text, padding="max_length", truncation=True, max_length=1024)
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized

dataset = dataset.map(tokenize)

# Load model with QLoRA
print("🧠 Loading model with LoRA...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Training arguments (no eval, enable checkpoints)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    max_steps=1000,
    learning_rate=1e-4,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    evaluation_strategy="no",  # ✅ Disable evaluation
    fp16=True,
    report_to="wandb",
    run_name="starcoderbase-finetune"
)

# Trainer (no eval dataset or callbacks)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# Train the model
print("🚀 Training...")
train_output = trainer.train()

# Save model and tokenizer
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved at {OUTPUT_DIR}")

# Optional: Save and plot loss graph
training_logs = trainer.state.log_history
steps = []
train_loss = []

for log in training_logs:
    if "loss" in log and "step" in log:
        train_loss.append(log["loss"])
        steps.append(log["step"])

if train_loss:
    plt.figure(figsize=(10, 5))
    plt.plot(steps, train_loss, label="Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Steps")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "train_loss.png"))
    plt.show()
    print(f"Training loss graph saved at {OUTPUT_DIR}/train_loss.png")
else:
    print("No loss data to plot.")
