from datasets import load_dataset
import pandas as pd


from datasets import load_dataset

ds = load_dataset("asoria/children-stories-dataset")
# Suppose it has a 'train' split:
df = ds["train"].to_pandas()

# Save to CSV
df.to_csv("stories.csv", index=False, encoding='utf-8-sig')
