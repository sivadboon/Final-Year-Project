from datasets import load_dataset, Dataset
from huggingface_hub import login
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Get Hugging Face token from .env
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN not found in .env file")

# Authenticate
login(token=HUGGINGFACE_TOKEN)

# Load and process StackExchange dataset
print("Loading dataset (streaming)...")
streaming_dataset = load_dataset("ArmelR/stack-exchange-instruction", split="test", streaming=True)

# Take 10,000 examples and convert to a regular Dataset
print("Converting to regular Dataset...")
examples = list(streaming_dataset.take(10000))
dataset = Dataset.from_list(examples)

# Save processed dataset
print("Saving preprocessed dataset...")
processed_path = "fine_tuning/processed_dataset"
dataset.save_to_disk(processed_path)
print(f"Dataset saved at {processed_path}")
