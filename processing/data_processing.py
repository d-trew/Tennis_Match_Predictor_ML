import pandas as pd
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ======================
# 1. Configuration
# ======================
DATA_DIR = "tennis_atp"
OUTPUT_FILE = "processed_tennis_data.csv"

print("\n=== Tennis Data Processing ===")
print(f"Data directory: {DATA_DIR}")
print(f"Output file: {OUTPUT_FILE}\n")

# ======================
# 2. Load Data
# ======================
print("1. Loading data...")

# Load all match files
all_matches = []
match_files = [f for f in os.listdir(DATA_DIR) if f.startswith("atp_matches_2") and f.endswith(".csv")]
print(f"Found {len(match_files)} match files")

for file in match_files:
    df = pd.read_csv(os.path.join(DATA_DIR, file), low_memory=False)
    print(f"  - Loaded {file}: {df.shape[0]} matches, {df.shape[1]} columns")
    all_matches.append(df)

matches = pd.concat(all_matches, ignore_index=True)
print(f"\nCombined matches: {matches.shape[0]} total matches, {matches.shape[1]} columns")
print("Sample columns:", matches.columns.tolist()[:10], "...")

# Load players data
print("\nLoading player data...")
players = pd.read_csv(
    os.path.join(DATA_DIR, "atp_players.csv"),
    dtype={
        'player_id': 'int32',
        'hand': 'category',
        'ioc': 'category',
        'height': 'float32',
        'wikidata_id': 'str'
    }
)
# Convert DOB column manually with proper format
players['dob'] = pd.to_datetime(players['dob'], format='%Y%m%d', errors='coerce')

# Remove players with impossible DOBs (born after dataset start or before 1900)
players = players[(players['dob'] < '2000-01-01') & (players['dob'] > '1900-01-01')]
print(f"Loaded {players.shape[0]} players with columns: {players.columns.tolist()}")
print(f"Date range of player DOBs: {players['dob'].min().year} to {players['dob'].max().year}")

# Load rankings data
print("\nLoading ranking data...")
all_rankings = []
ranking_files = [f for f in os.listdir(DATA_DIR) if (f.startswith("atp_rankings_0") or f.startswith("atp_rankings_1") or f.startswith("atp_rankings_2")) and f.endswith(".csv")]
print(f"Found {len(ranking_files)} ranking files")

for file in ranking_files:
    df = pd.read_csv(os.path.join(DATA_DIR, file),
                    dtype={
                        'rank': 'int16',
                        'player': 'int32',
                        'points': 'float32'
                    })
    # Convert date column manually to avoid warnings
    df['ranking_date'] = pd.to_datetime(df['ranking_date'].astype(str), format='%Y%m%d')
    all_rankings.append(df)

rankings = pd.concat(all_rankings, ignore_index=True)
print(f"\nCombined rankings: {rankings.shape[0]} records")
print(f"Date range: {rankings['ranking_date'].min()} to {rankings['ranking_date'].max()}")

# ======================
# 3. Data Cleaning
# ======================
print("\n2. Cleaning data...")

original_count = matches.shape[0]
matches = matches[~matches['score'].str.contains('RET|W/O', na=False)]
removed_count = original_count - matches.shape[0]
print(f"  - Removed {removed_count} incomplete matches (RET/W/O)")
print(f"  - Remaining matches: {matches.shape[0]}")

# Clean player data
print("\nProcessing player information...")
players['full_name'] = players['name_first'].str.strip() + ' ' + players['name_last'].str.strip()
players = players[['player_id', 'full_name', 'hand', 'dob', 'ioc', 'height']]
print(f"  - Created full_name column combining first and last names")
print(f"  - Final player columns: {players.columns.tolist()}")

# ======================
# 4. Feature Engineering
# ======================
print("\n3. Creating features...")

# Convert tourney_date to datetime with proper format
matches['tourney_date'] = pd.to_datetime(matches['tourney_date'].astype(str), format='%Y%m%d')
print(f"  - Converted tourney_date to datetime (range: {matches['tourney_date'].min()} to {matches['tourney_date'].max()}")

# Create player dictionary
player_dict = players.set_index('player_id').to_dict('index')
print(f"  - Created player lookup dictionary with {len(player_dict)} entries")

# Add player attributes
print("\nAdding player attributes to matches...")
for role in ['winner', 'loser']:
    # Add basic attributes
    matches[f'{role}_hand'] = matches[f'{role}_id'].map(lambda x: player_dict.get(x, {}).get('hand'))
    matches[f'{role}_height'] = matches[f'{role}_id'].map(lambda x: player_dict.get(x, {}).get('height'))
    
    # Get DOB from player_dict and ensure it's datetime
    dob_series = matches[f'{role}_id'].map(lambda x: player_dict.get(x, {}).get('dob'))
    matches[f'{role}_dob'] = pd.to_datetime(dob_series, errors='coerce')
    
    # Calculate age carefully - ensure both are datetime and clip to reasonable range
    matches[f'{role}_age'] = (
        (matches['tourney_date'] - matches[f'{role}_dob']).dt.days / 365.25
    ).clip(lower=12, upper=50).round(1)  # Reasonable tennis age range
    
    print(f"  - Added {role} attributes: hand, height, dob, age")
    print(f"    Age range for {role}s: {matches[f'{role}_age'].min():.1f} to {matches[f'{role}_age'].max():.1f} years")
    print(f"    Missing DOB for {matches[f'{role}_dob'].isna().sum()} {role}s")

    # Fill missing ages with median age by year
    median_age_by_year = matches.groupby(matches['tourney_date'].dt.year)[f'{role}_age'].transform('median')
    matches[f'{role}_age'] = matches[f'{role}_age'].fillna(median_age_by_year)
    print(f"    After filling missing: {matches[f'{role}_age'].isna().sum()} missing ages remaining")

# Sort matches by date for rolling calculations
matches = matches.sort_values('tourney_date')

# Calculate surface experience PROPERLY (pre-match counts)
print("\nCalculating surface experience (proper pre-match counts)...")
for surface in ['Hard', 'Clay', 'Grass']:
    # Create temporary copy sorted by date
    temp_matches = matches.sort_values('tourney_date').copy()
    
    # Calculate cumulative wins by surface up to but not including current match
    for role in ['winner', 'loser']:
        col_name = f'{role}_{surface.lower()}_exp'
        
        # Initialize with 0
        matches[col_name] = 0
        
        # For each player, calculate their surface wins before each match
        for player_id, player_matches in temp_matches.groupby(f'{role}_id'):
            # Get surface wins before each match
            player_matches = player_matches.sort_values('tourney_date')
            surface_wins = (player_matches['surface'] == surface).shift(1).cumsum().fillna(0)
            
            # Update in main dataframe
            matches.loc[player_matches.index, col_name] = surface_wins.values
            
    print(f"  - Calculated proper {surface} experience counts")

# Fix Recent Performance Calculation
print("\nCalculating recent performance (fixed implementation)...")

def calculate_recent_wins(df, player_col='winner_id'):
    """Calculate wins in last 180 days for each player before each match"""
    df = df.sort_values('tourney_date')
    result = pd.Series(0, index=df.index, dtype='int32')
    
    for player_id, player_matches in df.groupby(player_col):
        # Calculate days since each match
        player_matches = player_matches.sort_values('tourney_date')
        dates = player_matches['tourney_date']
        
        # For each match, count wins in previous 180 days
        for i, current_date in enumerate(dates):
            start_date = current_date - pd.Timedelta(days=180)
            win_count = ((dates.iloc[:i] >= start_date) & 
                        (dates.iloc[:i] < current_date)).sum()
            result.at[player_matches.index[i]] = win_count
    
    return result

matches['winner_recent_wins'] = calculate_recent_wins(matches, 'winner_id')
matches['loser_recent_wins'] = calculate_recent_wins(matches, 'loser_id')

print("  - Recent wins calculation complete")
print(f"    Winner recent wins stats: Min={matches['winner_recent_wins'].min()}, Max={matches['winner_recent_wins'].max()}")
print(f"    Loser recent wins stats: Min={matches['loser_recent_wins'].min()}, Max={matches['loser_recent_wins'].max()}")

# Ranking features
print("\nProcessing ranking data...")
rankings = rankings.sort_values(['player', 'ranking_date'])

def get_rank_points(player_id, date):
    """Get most recent ranking points before given date"""
    try:
        player_rankings = rankings[rankings['player'] == player_id]
        return player_rankings[player_rankings['ranking_date'] <= date].iloc[-1]['points']
    except:
        return None

if 'winner_rank_points' not in matches.columns:
    print("  - Calculating ranking points (this may take a while)...")
    matches['winner_rank_points'] = matches.apply(
        lambda row: get_rank_points(row['winner_id'], row['tourney_date']), axis=1)
    matches['loser_rank_points'] = matches.apply(
        lambda row: get_rank_points(row['loser_id'], row['tourney_date']), axis=1)
    print("  - Ranking points calculation complete")

# Calculate and cap rank points ratio
matches['rank_points_ratio'] = matches['winner_rank_points'] / matches['loser_rank_points'].replace(0, 1)
matches['rank_points_ratio'] = matches['rank_points_ratio'].clip(upper=20)  # Cap at 20
print(f"  - Calculated rank_points_ratio (min: {matches['rank_points_ratio'].min():.2f}, max: {matches['rank_points_ratio'].max():.2f})")

# Head-to-head features
print("\nCalculating head-to-head records...")
h2h = matches.groupby(['winner_id', 'loser_id']).size().reset_index(name='h2h_wins')
matches = matches.merge(
    h2h,
    left_on=['winner_id', 'loser_id'],
    right_on=['winner_id', 'loser_id'],
    how='left'
).fillna(0)
print(f"  - Added h2h_wins column (max encounters: {matches['h2h_wins'].max()})")

# ======================
# 5. Data Validation
# ======================
print("\nRunning data validation checks...")

# Validate age ranges
print("\nAge Validation:")
print("Winner age distribution:")
print(matches['winner_age'].describe())
print("\nLoser age distribution:")
print(matches['loser_age'].describe())

# Validate experience counts
print("\nExperience Validation:")
for surface in ['hard', 'clay', 'grass']:
    print(f"\nMax {surface} experience:")
    print(f"Winners: {matches[f'winner_{surface}_exp'].max()}")
    print(f"Losers: {matches[f'loser_{surface}_exp'].max()}")

# Validate recent wins
print("\nRecent Wins Validation:")
print("Winner recent wins distribution:")
print(matches['winner_recent_wins'].describe())
print("\nLoser recent wins distribution:")
print(matches['loser_recent_wins'].describe())

# ======================
# 6. Save Processed Data
# ======================
print("\n4. Saving results...")

output_cols = [
    'tourney_date', 'surface', 'tourney_level', 'round',
    'winner_id', 'winner_name', 'winner_hand', 'winner_age', 'winner_height',
    'winner_rank', 'winner_rank_points', 'winner_recent_wins',
    'loser_id', 'loser_name', 'loser_hand', 'loser_age', 'loser_height',
    'loser_rank', 'loser_rank_points', 'loser_recent_wins',
    'winner_hard_exp', 'winner_clay_exp', 'winner_grass_exp',
    'loser_hard_exp', 'loser_clay_exp', 'loser_grass_exp',
    'rank_points_ratio', 'h2h_wins'
]

print(f"\nFinal dataset columns ({len(output_cols)} total):")
for i, col in enumerate(output_cols, 1):
    print(f"{i:2d}. {col}")

matches[output_cols].to_csv(OUTPUT_FILE, index=False)
print(f"\nProcessing complete! Saved {matches.shape[0]} matches to {OUTPUT_FILE}")
print("=== End of processing ===")