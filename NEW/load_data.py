from datasets import load_dataset
import pandas as pd


# Load only the first 3000 rows
ds = load_dataset(
    "ajibawa-2023/Children-Stories-Collection",
    split="train[:1000]"
)
# Suppose it has a 'train' split:
df = ds["train"].to_pandas()

# Save to CSV
df.to_csv("stories.csv", index=False, encoding='utf-8-sig')
