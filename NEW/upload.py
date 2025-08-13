import pandas as pd
from datasets import Dataset
from huggingface_hub import login
import os

# -------------------------
# 1. Config
# -------------------------
CSV_FILE = "final_data.csv"  # your local CSV file
HF_DATASET_NAME = "Shakil2448868/Bangla-Stories"  # Change to your HF repo name
HF_TOKEN = os.getenv("HF_TOKEN")  # or put your token here directly

# -------------------------
# 2. Login to HF Hub
# -------------------------
if not HF_TOKEN:
    HF_TOKEN = input("Enter your Hugging Face token: ").strip()
login(token=HF_TOKEN)

# -------------------------
# 3. Load and clean CSV
# -------------------------
df = pd.read_csv(CSV_FILE)

# Keep only desired columns
df = df[["text", "trans_bangla"]]

# Reset index
df.reset_index(drop=True, inplace=True)

# -------------------------
# 4. Convert to HF Dataset
# -------------------------
dataset = Dataset.from_pandas(df)

# -------------------------
# 5. Push to Hugging Face Hub
# -------------------------
dataset.push_to_hub(HF_DATASET_NAME)

print(f"✅ Dataset uploaded to https://huggingface.co/datasets/{HF_DATASET_NAME}")
