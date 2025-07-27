import pandas as pd

# Read the original CSV file
df = pd.read_csv('processed_tennis_data.csv')

# Filter matches where surface is 'Clay'
clay_matches = df[df['surface'] == 'Clay']

# Save to a new CSV file
clay_matches.to_csv('clay_matches.csv', index=False)

print(f"Saved {len(clay_matches)} clay court matches to 'clay_matches.csv'.")