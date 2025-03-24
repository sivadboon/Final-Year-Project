# 🤖 Fine-Tuned StarCoder for AI-Assisted Programming

This project fine-tunes the [StarCoderBase-7B](https://huggingface.co/bigcode/starcoderbase) model using QLoRA for efficient code generation. It supports instruction-following code synthesis in Python and is deployable via a lightweight Flask web app.

## 🚀 Project Highlights

- 📚 Fine-tuned on Stack Exchange Instruction Dataset
- ⚡ QLoRA + LoRA for efficient fine-tuning on a 12GB GPU
- 🔗 Adapter merged into the base model for seamless inference
- 🌐 Flask-based web interface with Docker support
- 🧪 Evaluated on HumanEval with competitive results

---

## 📁 Downloads

| Folder | Description | Download Link |
|--------|-------------|----------------|
| `fine_tuned_model` | LoRA adapter weights and config files | [📥 Download](https://drive.google.com/uc?export=download&id=1_acihG8SIMAUoU3KjRCXXMq4oCM35UuQ) |
| `merged_model` | StarCoderBase-7B with LoRA merged in | [📥 Download](https://drive.google.com/uc?export=download&id=1q-OCGOujT1yz7163C_3VBbicaHMab-1y) |
| `preprocessed_data` | Training data subset from Stack Exchange (tokenized) | [📥 Download](https://drive.google.com/uc?export=download&id=1rBpVtU7WrFl1Xswf11rzbbRqybgIFepN) |


> ✅ *All links initiate direct file downloads from Google Drive.*

---

## 🧠 Model Details

- **Base Model**: `bigcode/starcoderbase-7b`
- **Fine-Tuning Method**: QLoRA (4-bit) with LoRA
- **Dataset**: 10K examples from [`stack-exchange-instruction`](https://huggingface.co/datasets/ArmelR/stack-exchange-instruction)
- **Training Framework**: Hugging Face Transformers + PEFT + bitsandbytes

---

## 🧪 Evaluation Results (HumanEval)

| Model | Pass@1 | Pass@5 |
|-------|--------|--------|
| GPT-4 (via API) | 0.1037 | 0.2866 |
| CodeLlama-7B-Python | 0.0439 | 0.1707 |
| StarCoderBase-7B | 0.5183 | 0.9024 |
| **Fine-Tuned StarCoder-7B** | **0.5780** | **0.9146** |

> 📊 The fine-tuned model shows a notable gain over the base, especially in one-shot accuracy (pass@1).

---

## 💻 Deployment Instructions

### 1. Clone and Build

- Download the fine_tuned_model and merged_model and put it in the Final Year Project Directory

```bash
git clone https://github.com/sivadboon/your-repo-name.git
cd your-repo-name
docker build -t starcoder-flask-app .
```

### 2. Run the Docker Container

```bash
docker run --gpus all -p 5000:5000 starcoder-flask-app
```

### 3. Open the Webpage

- You must have a good GPU to use this application
