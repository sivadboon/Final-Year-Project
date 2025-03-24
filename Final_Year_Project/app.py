from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
import re

app = Flask(__name__)
MODEL_PATH = "merged_model"

def load_model():
    try:
        print("Loading model with device_map=auto")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load on GPU: {e}")
        print("Falling back to CPU.")
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        return model, tokenizer

model, tokenizer = load_model()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    prompt = ""
    user_input = ""  

    if request.method == "POST":
        user_input = request.form["prompt"].strip()
        prompt = f"### Task: {user_input}\n```python\n"

        print(f"🚀 Prompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only first Python code block
        code_match = re.search(r"```python(.*?)```", generated, re.DOTALL)
        if code_match:
            result = code_match.group(1).strip()
        else:
            # fallback: remove prompt prefix if regex fails
            result = generated[len(prompt):].strip()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return render_template("index.html", result=result, prompt=user_input)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
