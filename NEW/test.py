import pandas as pd

# Read your CSV
df = pd.read_csv("final_song_output.csv")

# Drop 'syn_prompt', keep only required columns, and rename 'new_song'
new_df = df.drop(columns=["syn_prompt"])[["Writer", "Title", "Song", "new_song"]].rename(
    columns={"new_song": "newly_generated_song"}
)

# Save to CSV
new_df.to_csv("final_song_cleaned.csv", index=False, encoding="utf-8-sig")

