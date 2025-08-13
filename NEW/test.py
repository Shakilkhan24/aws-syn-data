import pandas as pd

# Read your CSV
df = pd.read_csv("batch_1100.csv")

# Keep only rows where 'trans_bangla' is not NaN
df_filtered = df[df["trans_bangla"].notna()]

# Save to a new CSV
df_filtered.to_csv("final_data.csv", index=False)

print(f"Saved {len(df_filtered)} rows to new.csv")
