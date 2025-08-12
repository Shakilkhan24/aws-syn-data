from datasets import load_dataset
import pandas as pd

from datasets import load_dataset

# Load only the first 3000 rows
ds = load_dataset(
    "ajibawa-2023/Children-Stories-Collection",
    split="train[:10000]"
)
print(ds)
# Suppose it has a 'train' split:
df = ds.to_pandas()

# Save to CSV
df.to_csv("story_collection.csv", index=False, encoding='utf-8-sig')
