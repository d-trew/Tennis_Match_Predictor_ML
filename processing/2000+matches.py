import os

import pandas as pd


DATA_DIR = "tennis_atp"
all_matches = []
match_files = [f for f in os.listdir(DATA_DIR) if f.startswith("atp_matches_2") and f.endswith(".csv")]
print(f"Found {len(match_files)} match files")

for file in match_files:
    df = pd.read_csv(os.path.join(DATA_DIR, file), low_memory=False)
    print(f"  - Loaded {file}: {df.shape[0]} matches, {df.shape[1]} columns")
    all_matches.append(df)

matches = pd.concat(all_matches, ignore_index=True)
print(f"\nCombined matches: {matches.shape[0]} total matches, {matches.shape[1]} columns")

matches.to_csv("tennis_atp_2000_plus_matches.csv", index=False)