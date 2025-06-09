import pandas as pd
import matplotlib.pyplot as plt

# Đọc file CSV
df = pd.read_csv("metrics_full.csv")

# Biểu đồ Loss
plt.figure(figsize=(10, 4))
plt.plot(df["step"], df["train_loss"], label="Train Loss")
plt.plot(df["step"], df["val_loss"], label="Val Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")

# Biểu đồ Perplexity
plt.figure(figsize=(10, 4))
plt.plot(df["step"], df["train_ppl"], label="Train PPL")
plt.plot(df["step"], df["val_ppl"], label="Val PPL")
plt.xlabel("Step")
plt.ylabel("Perplexity")
plt.title("Training and Validation Perplexity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ppl_curve.png")

# Biểu đồ Accuracy
plt.figure(figsize=(10, 4))
plt.plot(df["step"], df["train_acc"], label="Train Accuracy")
plt.plot(df["step"], df["val_acc"], label="Val Accuracy")
plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("acc_curve.png")
