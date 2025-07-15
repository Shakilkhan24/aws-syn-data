from datasets import load_dataset
import pandas as pd

# Load the dataset
ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")

# If it's a DatasetDict, pick the split (e.g., 'train')
# Many datasets have 'train', 'test', 'validation' splits.
print(ds)  # Check what splits are available!

# Suppose it has a 'train' split:
df = ds["train"].to_pandas()

# Save to CSV
df.to_csv("medical.csv", index=False, encoding='utf-8-sig')
