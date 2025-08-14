from datasets import load_dataset
import pandas as pd

# Load dataset
ds = load_dataset("Shakil2448868/bangla-songs-synthetic-prompt")

print(ds)  # This will show available splits (e.g., 'train')

# Convert the 'train' split to pandas DataFrame
df = ds["train"].to_pandas()

# Save to CSV
df.to_csv("song.csv", index=False, encoding='utf-8-sig')
