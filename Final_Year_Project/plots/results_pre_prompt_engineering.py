import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data from Table 5.1
models = [
    "Merged Fine-Tuned StarCoder",
    "StarCoderBase-7B",
    "GPT-4 (API)",
    "CodeLlama-7B-Instruct",
    "CodeLlama-7B-Python",
    "CodeLlama-7B (Base)"
]

pass_1 = [0.578, 0.518, 0.104, 0.016, 0.044, 0.013]
pass_5 = [0.915, 0.902, 0.287, 0.079, 0.171, 0.067]

# Create a DataFrame for clarity
df = pd.DataFrame({
    'Model': models,
    'pass@1': pass_1,
    'pass@5': pass_5
})

# Sort models by pass@1 for clearer visualization
df_sorted = df.sort_values('pass@1', ascending=False)

# Bar Chart
fig1, ax1 = plt.subplots(figsize=(10, 6))
x = np.arange(len(df_sorted))
bar_width = 0.35
ax1.bar(x - bar_width/2, df_sorted['pass@1'], bar_width, label='pass@1')
ax1.bar(x + bar_width/2, df_sorted['pass@5'], bar_width, label='pass@5')
ax1.set_xlabel('Model')
ax1.set_ylabel('Accuracy')
ax1.set_title('Comparison of pass@1 and pass@5 on HumanEval')
ax1.set_xticks(x)
ax1.set_xticklabels(df_sorted['Model'], rotation=45, ha='right')
ax1.legend()
plt.tight_layout()
plt.show()


