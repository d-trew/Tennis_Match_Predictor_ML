import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import optuna
import pickle
import time
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import StackingClassifier
from collections import defaultdict
import os
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TennisHyperparameterTuner:
    def __init__(self, X, y, dates):
        self.X = X
        self.y = y
        self.dates = pd.to_datetime(dates)
        self.tscv = TimeSeriesSplit(n_splits=5)

    def objective(self, trial):
        params = {
            'xgb': {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 300, 800),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
                # 'colsample_bytree': trial.suggest_float('xgb_colsample', 0.6, 1.0),
                'gamma': trial.suggest_float('xgb_gamma', 0, 0.5)
            },
            'lgb': {
                'n_estimators': trial.suggest_int('lgb_n_estimators', 300, 800),
                'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('lgb_lr', 0.01, 0.2, log=True),
                'feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('lgb_bagging_fraction', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('lgb_min_child', 10, 50)
            },
            'cat': {
                'iterations': trial.suggest_int('cat_iterations', 300, 800),
                'depth': trial.suggest_int('cat_depth', 4, 10),
                'learning_rate': trial.suggest_float('cat_lr', 0.01, 0.2, log=True),
                'l2_leaf_reg': trial.suggest_float('cat_l2', 1, 10),
                'random_strength': trial.suggest_float('cat_random_strength', 0.1, 1.0)
            }
        }

        # Time-series cross-validation
        auc_scores = []
        for train_idx, val_idx in self.tscv.split(self.X):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

            models = {
                'xgb': xgb.XGBClassifier(**params['xgb'], random_state=42),
                'lgb': lgb.LGBMClassifier(**params['lgb'], random_state=42, verbose=-1),
                'cat': CatBoostClassifier(**params['cat'], random_seed=42, verbose=False)
            }

            stack = StackingClassifier(
                estimators=list(models.items()),
                final_estimator=LogisticRegression(C=0.1, max_iter=1000),
                cv=5,
                n_jobs=-1
            )

            stack.fit(X_train, y_train)
            preds = stack.predict_proba(X_val)[:, 1]
            auc_scores.append(roc_auc_score(y_val, preds))

        return np.mean(auc_scores)

    def optimize(self, n_trials=10):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_params


class TennisDataProcessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.features = []
        self.player_stats = {}
        self.h2h_stats = {}

    def load_and_clean_data(self, filepath):
        """Load and clean the tennis data"""
        print("🎾 Loading tennis data...")

        # Load data
        df = pd.read_csv(filepath)
        print(f"   📊 Loaded {len(df)} matches")

        # Convert date
        df['tourney_date'] = pd.to_datetime(
            df['tourney_date'], format='%Y%m%d')

        # Clean missing values in critical columns
        critical_cols = ['winner_rank', 'loser_rank',
            'winner_rank_points', 'loser_rank_points']
        for col in critical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())

        # Handle missing match statistics (fill with player averages or 0)
        stat_cols = ['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
                    'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'minutes']

        for col in stat_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Sort by date for temporal features
        df = df.sort_values('tourney_date').reset_index(drop=True)

        print(
            f"   ✅ Data cleaned. Date range: {df['tourney_date'].min()} to {df['tourney_date'].max()}")
        return df

    def advanced_missing_value_handling(self, X):
        """More sophisticated missing value treatment"""
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer

        print("   🔄 Using iterative imputation for missing values...")
        imputer = IterativeImputer(random_state=42, max_iter=10)
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        return X_imputed

    def calculate_historical_stats(self, df, cache_file="player_history_cached.csv"):
        """Calculate historical performance stats for each player"""

        if os.path.exists(cache_file):
            print(f"📂 Loading cached stats from {cache_file}...")
            return pd.read_csv(cache_file)

        print("📈 Calculating historical player statistics...")

        # Initialize player tracking
        player_history = {}
        h2h_stats = defaultdict(
            lambda: {'matches': [], 'p1_wins': 0, 'p2_wins': 0})
        processed_rows = []
        print(" columns: ", df.columns)
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"   Processing match {idx}/{len(df)}")

            winner_id = row['winner_id']
            loser_id = row['loser_id']
            match_date = row['tourney_date']
            surface = row['surface']
            match_id = f"{row['tourney_id']}_{row['match_num']}"

            # Get historical stats for both players UP TO this match date
            winner_stats = self.get_player_stats_before_date(
                player_history, winner_id, match_date, surface)
            loser_stats = self.get_player_stats_before_date(
                player_history, loser_id, match_date, surface)

            # Create enhanced row with historical features
            enhanced_row = row.copy()

            enhanced_row['match_id'] = match_id

            # Add historical features
            for stat_name, stat_value in winner_stats.items():
                enhanced_row[f'winner_{stat_name}'] = stat_value

            for stat_name, stat_value in loser_stats.items():
                enhanced_row[f'loser_{stat_name}'] = stat_value

            # --- Head-to-head feature calculation
            key = tuple(sorted([winner_id, loser_id]))
            stats = h2h_stats[key]
            total_matches = len(stats['matches'])

            # Determine if winner is p1 in sorted key
            if total_matches == 0:
                win_rate = 0.5
            else:
                if winner_id == key[0]:
                    win_rate = stats['p1_wins'] / total_matches
                else:
                    win_rate = stats['p2_wins'] / total_matches

            # Add H2H features to row
            enhanced_row['h2h_matches'] = total_matches
            enhanced_row['h2h_win_rate'] = win_rate

            # Save before updating history
            processed_rows.append(enhanced_row)

            # Update player history AFTER using it for prediction
            self.update_player_history(
                player_history, winner_id, row, 'winner', match_date, surface)
            self.update_player_history(
                player_history, loser_id, row, 'loser', match_date, surface)

            # --- Update H2H stats AFTER recording features
            stats['matches'].append(match_date)
            if winner_id == key[0]:
                stats['p1_wins'] += 1
            else:
                stats['p2_wins'] += 1
            h2h_stats[key] = stats

            if idx < 5 or idx > len(df) - 5:
                print(f"[DEBUG] Match {match_id}: {winner_id} vs {loser_id} | "
                    f"H2H Matches: {total_matches} | Win Rate: {win_rate:.2f}")

        enhanced_df = pd.DataFrame(processed_rows)
        # Save to CSV for future use
        enhanced_df.to_csv(cache_file, index=False)
        print(f"💾 Saved cached stats to {cache_file}")
        return enhanced_df

    def get_player_stats_before_date(self, player_history, player_id, match_date, surface):
        """Get player statistics before a specific date"""
        if player_id not in player_history:
            return {
                'matches_played': 0,
                'win_rate': 0.5,
                'surface_matches': 0,
                'surface_win_rate': 0.5,
                'recent_form_10': 0.5,
                'avg_rank': 100,
                'days_since_last': 365,
                'ace_rate': 0.0,
                'df_rate': 0.0,
                'first_serve_pct': 0.65,
                'first_serve_won': 0.65,
                'break_points_saved': 0.6,
                'rank_points': 0,
                'momentum_score': 0.5,
                'rank_trend': 0.0,
                'peak_rank': 2500,
                'fatigue': 0.0
            }

        history = player_history[player_id]

        # Filter matches before this date
        relevant_matches = [m for m in history['matches']
            if m['date'] < match_date]

        if not relevant_matches:
            return {
                'matches_played': 0,
                'win_rate': 0.5,
                'surface_matches': 0,
                'surface_win_rate': 0.5,
                'recent_form_10': 0.5,
                'avg_rank': 100,
                'days_since_last': 365,
                'ace_rate': 0.0,
                'df_rate': 0.0,
                'first_serve_pct': 0.65,
                'first_serve_won': 0.65,
                'break_points_saved': 0.6,
                'rank_points': history.get('latest_rank_points', 0),
                'momentum_score': 0.5,
                'rank_trend': 0.0,
                'peak_rank': history.get('peak_rank', 2500),
                'fatigue': 0.0
            }

        # Calculate statistics
        total_matches = len(relevant_matches)
        wins = sum(1 for m in relevant_matches if m['won'])
        win_rate = wins / total_matches if total_matches > 0 else 0.5

        # Surface-specific stats
        surface_matches = [
            m for m in relevant_matches if m['surface'] == surface]
        surface_total = len(surface_matches)
        surface_wins = sum(1 for m in surface_matches if m['won'])
        surface_win_rate = surface_wins / surface_total if surface_total > 0 else 0.5

        # Recent form (last 10 matches)
        recent_matches = sorted(
            relevant_matches, key=lambda x: x['date'], reverse=True)[:10]
        recent_wins = sum(1 for m in recent_matches if m['won'])
        recent_form = recent_wins / \
            len(recent_matches) if recent_matches else 0.5

        # Average rank
        ranks = [m['rank'] for m in relevant_matches if m['rank'] > 0]
        avg_rank = np.mean(ranks) if ranks else 100

        # Days since last match
        last_match_date = max(m['date'] for m in relevant_matches)
        days_since = (match_date - last_match_date).days

        # Playing style stats
        ace_rates = [m.get('ace_rate', 0) for m in relevant_matches]
        df_rates = [m.get('df_rate', 0) for m in relevant_matches]
        first_serve_pcts = [m.get('first_serve_pct', 0.65)
                                  for m in relevant_matches]
        first_serve_wons = [m.get('first_serve_won', 0.65)
                                  for m in relevant_matches]
        bp_saveds = [m.get('bp_saved', 0.6) for m in relevant_matches]

        # Momentum score (weighted recent performance)
        momentum_weights = [0.4, 0.3, 0.2, 0.1] if len(recent_matches) >= 4 else [
                                                       1.0/len(recent_matches)] * len(recent_matches)
        momentum_score = sum(w * (1 if m['won'] else 0) for w, m in zip(
            momentum_weights, recent_matches[:4])) if recent_matches else 0.5

        # --- RANK TREND ---
        rank_history = player_history[player_id].get('rank_history', [])

        # --- Optimized Rank Trend Calculation ---
        rank_trend = 0.0
        rank_history = player_history[player_id].get('rank_history', [])

        if rank_history:
            df_ranks = pd.DataFrame(rank_history, columns=['date', 'rank'])
            df_ranks['date'] = pd.to_datetime(df_ranks['date'])

            # Filter to 6 months before match date
            cutoff = pd.to_datetime(match_date) - pd.Timedelta(days=180)
            df_recent = df_ranks[(df_ranks['date'] < pd.to_datetime(
                match_date)) & (df_ranks['date'] >= cutoff)]

            # Drop duplicate dates (keeping first occurrence)
            df_recent = df_recent.drop_duplicates(
                subset='date').sort_values('date')

            if len(df_recent) >= 3:
                days = (df_recent['date'] -
                        df_recent['date'].iloc[0]).dt.days.values
                values = df_recent['rank'].values

                if len(set(days)) > 1 and not np.isnan(days).any() and not np.isnan(values).any():
                    try:
                        slope = np.polyfit(days, values, 1)[0]
                        rank_trend = -slope  # Lower slope = improving
                    except np.linalg.LinAlgError:
                        rank_trend = 0.0

        # --- PEAK RANK ---
        peak_rank = player_history[player_id].get('peak_rank', 2500)

        # --- FATIGUE ---
        match_dates = player_history[player_id].get('match_dates', [])
        fatigue_matches = [
            d for d in match_dates
            if 0 < (pd.to_datetime(match_date) - pd.to_datetime(d)).days <= 14
        ]
        fatigue = len(fatigue_matches) / 5  # Normalize by 5 matches
        # -------------------

        if player_id == 'some_known_player':
            print(f"--- DEBUG for {player_id} on {match_date} ---")
            print("Total relevant matches:", len(relevant_matches))
            print("Rank history:", player_history[player_id].get(
                'rank_history', []))
            print("Match dates:", player_history[player_id].get(
                'match_dates', []))
            print("Peak rank:", player_history[player_id].get('peak_rank'))

        return {
            'matches_played': total_matches,
            'win_rate': win_rate,
            'surface_matches': surface_total,
            'surface_win_rate': surface_win_rate,
            'recent_form_10': recent_form,
            'avg_rank': avg_rank,
            'days_since_last': min(days_since, 365),  # Cap at 1 year
            'ace_rate': np.mean(ace_rates) if ace_rates else 0.0,
            'df_rate': np.mean(df_rates) if df_rates else 0.0,
            'first_serve_pct': np.mean(first_serve_pcts) if first_serve_pcts else 0.65,
            'first_serve_won': np.mean(first_serve_wons) if first_serve_wons else 0.65,
            'break_points_saved': np.mean(bp_saveds) if bp_saveds else 0.6,
            'rank_points': history.get('latest_rank_points', 0),
            'momentum_score': momentum_score,
            'rank_trend': rank_trend,
            'peak_rank': peak_rank,
            'fatigue': fatigue
        }

    def update_player_history(self, player_history, player_id, match_row, player_type, match_date, surface):
        """Update player history after a match"""
        if player_id not in player_history:
            player_history[player_id] = {
                'matches': [],
                'latest_rank_points': 0,
                'rank_history': [],
                'match_dates': [],
                'peak_rank': 2500
            }

        # Determine if won
        won = (player_type == 'winner')

        # Get match statistics
        prefix = 'w_' if player_type == 'winner' else 'l_'

        import math
        rank = match_row.get(f'{player_type}_rank')
        if rank is None or (isinstance(rank, float) and math.isnan(rank)):
            rank = match_row.get(f'{prefix}rank')
        if rank is None or (isinstance(rank, float) and math.isnan(rank)):
            rank = 100
            print(
                f"[Warning] Rank missing for {player_type} in match on {match_date} — defaulting to 100")

        if rank is None or (isinstance(rank, float) and math.isnan(rank)):
            print("DEBUG: match_row keys:", match_row.keys())
            print("DEBUG: match_row values:", match_row)
            print(
                f"[Warning] Rank missing for {player_type} in match on {match_date}")

        match_record = {
            'date': match_date,
            'surface': surface,
            'won': won,
            'rank': rank,
            'rank_points': match_row.get(f'{player_type}_rank_points', 0),
        }

        # Add playing style stats if available
        if f'{prefix}svpt' in match_row and match_row[f'{prefix}svpt'] > 0:
            match_record.update({
                'ace_rate': match_row.get(f'{prefix}ace', 0) / match_row[f'{prefix}svpt'],
                'df_rate': match_row.get(f'{prefix}df', 0) / match_row[f'{prefix}svpt'],
                'first_serve_pct': match_row.get(f'{prefix}1stIn', 0) / match_row[f'{prefix}svpt'],
                'first_serve_won': match_row.get(f'{prefix}1stWon', 0) / max(match_row.get(f'{prefix}1stIn', 1), 1),
                'bp_saved': match_row.get(f'{prefix}bpSaved', 0) / max(match_row.get(f'{prefix}bpFaced', 1), 1)
            })

        player_history[player_id]['matches'].append(match_record)
        player_history[player_id]['latest_rank_points'] = match_record['rank_points']

        # ✅ Update rank history
        player_history[player_id].setdefault(
            'rank_history', []).append((str(match_date), rank))

        # ✅ Update match dates for fatigue
        player_history[player_id].setdefault(
            'match_dates', []).append(match_date)

        # ✅ Update peak rank
        if 'peak_rank' not in player_history[player_id]:
            player_history[player_id]['peak_rank'] = rank
        else:
            player_history[player_id]['peak_rank'] = min(
                player_history[player_id]['peak_rank'], rank)

    def add_advanced_features(self, pred_df):
        """Add more sophisticated features"""
        print("🧠 Adding advanced features...")

        # ✅ Ranking momentum
        pred_df['rank_trend_diff'] = pred_df['player1_rank_trend'] - \
            pred_df['player2_rank_trend']
        # ✅ Surface expertise
        pred_df['surface_expertise_p1'] = pred_df['player1_surface_win_rate'] - \
            pred_df['player1_win_rate']
        pred_df['surface_expertise_p2'] = pred_df['player2_surface_win_rate'] - \
            pred_df['player2_win_rate']
        pred_df['surface_expertise_diff'] = pred_df['surface_expertise_p1'] - \
            pred_df['surface_expertise_p2']

        # ✅ Fatigue
        pred_df['fatigue_diff'] = pred_df['player1_fatigue'] - \
            pred_df['player2_fatigue']

        # ✅ Peak performance
        pred_df['peak_rank_diff'] = pred_df['player1_peak_rank'] - \
            pred_df['player2_peak_rank']

        print(pred_df[['player1_rank_trend',
              'player2_rank_trend', 'rank_trend_diff']].describe())
        print(
            pred_df[['player1_fatigue', 'player2_fatigue', 'fatigue_diff']].describe())
        print(pred_df[['player1_peak_rank', 'player2_peak_rank',
              'peak_rank_diff']].describe())
        print(pred_df[['surface_expertise_diff']].describe())

        # NEW FEATURE COMBINATIONS - Made performance worse

        print("✅ Advanced features added.")

        return pred_df

    def create_prediction_features(self, df):
        """Create features for prediction (NO DATA LEAKAGE)"""
        print("🔧 Creating prediction features...")
        round_mapping = {
        'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4,
        'QF': 5, 'SF': 6, 'F': 7, 'RR': 3,  # Round Robin matches as R32 level
        'BR': 3, 'ER': 1  # Qualifying rounds
        }
        # Convert to player1 vs player2 format to eliminate data leakage
        prediction_data = []

        for _, row in df.iterrows():
            # Get H2H stats BEFORE creating any examples for this match
            p1_id = row.get('winner_id', 'unknown')
            p2_id = row.get('loser_id', 'unknown')
            key = tuple(sorted([p1_id, p2_id]))

            if key not in self.h2h_stats:
                self.h2h_stats[key] = {
                    'matches': [], 'p1_wins': 0, 'p2_wins': 0}

            h2h_data = self.h2h_stats[key]
            total_h2h_matches = len(h2h_data['matches'])

            # Calculate H2H win rates for both players
            if total_h2h_matches == 0:
                winner_h2h_rate = 0.5
                loser_h2h_rate = 0.5
            else:
                if p1_id == key[0]:  # winner is first in sorted key
                    winner_h2h_rate = h2h_data['p1_wins'] / total_h2h_matches
                    loser_h2h_rate = h2h_data['p2_wins'] / total_h2h_matches
                else:  # winner is second in sorted key
                    winner_h2h_rate = h2h_data['p2_wins'] / total_h2h_matches
                    loser_h2h_rate = h2h_data['p1_wins'] / total_h2h_matches

            round_num = round_mapping.get(
                row['round'], 3)  # Default to R32 level

            # Create two examples: one from each player's perspective

            # Player 1 perspective (originally winner)
            example1 = {
                'match_id': f"{row['tourney_id']}_{row['match_num']}_1",
                'tourney_date': row['tourney_date'],
                'surface': row['surface'],
                'tourney_level': row['tourney_level'],
                'draw_size': row['draw_size'],
                'round': row['round'],
                'best_of': row.get('best_of', 3),

                # Player features (avoid winner/loser terminology)
                'player1_id': p1_id,
                'player2_id': p2_id,
                'player1_rank': row.get('winner_rank', 100),
                'player2_rank': row.get('loser_rank', 100),
                'player1_rank_points': row.get('winner_rank_points', 0),
                'player2_rank_points': row.get('loser_rank_points', 0),
                'player1_age': row.get('winner_age', 25),
                'player2_age': row.get('loser_age', 25),

                # Historical stats
                'player1_matches_played': row.get('winner_matches_played', 0),
                'player2_matches_played': row.get('loser_matches_played', 0),
                'player1_win_rate': row.get('winner_win_rate', 0.5),
                'player2_win_rate': row.get('loser_win_rate', 0.5),
                'player1_surface_win_rate': row.get('winner_surface_win_rate', 0.5),
                'player2_surface_win_rate': row.get('loser_surface_win_rate', 0.5),
                'player1_recent_form': row.get('winner_recent_form_10', 0.5),
                'player2_recent_form': row.get('loser_recent_form_10', 0.5),
                'player1_momentum': row.get('winner_momentum_score', 0.5),
                'player2_momentum': row.get('loser_momentum_score', 0.5),
                'player1_days_since': row.get('winner_days_since_last', 30),
                'player2_days_since': row.get('loser_days_since_last', 30),

                # H2H features (calculated before this match)
                'h2h_matches': total_h2h_matches,
                'h2h_win_rate': winner_h2h_rate,

                # Additional features
                'player1_rank_trend': row.get('winner_rank_trend', 0.0),
                'player2_rank_trend': row.get('loser_rank_trend', 0.0),
                'player1_fatigue': row.get('winner_fatigue', 0.0),
                'player2_fatigue': row.get('loser_fatigue', 0.0),
                'player1_peak_rank': row.get('winner_peak_rank', 2500),
                'player2_peak_rank': row.get('loser_peak_rank', 2500),
                'round_numeric': round_num,

                # Target (1 if player1 wins, 0 if player2 wins)
                'target': 1
            }

            # Player 2 perspective (originally loser)
            example2 = {
                'match_id': f"{row['tourney_id']}_{row['match_num']}_2",
                'tourney_date': row['tourney_date'],
                'surface': row['surface'],
                'tourney_level': row['tourney_level'],
                'draw_size': row['draw_size'],
                'round': row['round'],
                'best_of': row.get('best_of', 3),

                # Flip player roles
                'player1_id': p2_id,
                'player2_id': p1_id,
                'player1_rank': row.get('loser_rank', 100),
                'player2_rank': row.get('winner_rank', 100),
                'player1_rank_points': row.get('loser_rank_points', 0),
                'player2_rank_points': row.get('winner_rank_points', 0),
                'player1_age': row.get('loser_age', 25),
                'player2_age': row.get('winner_age', 25),

                # Flip historical stats
                'player1_matches_played': row.get('loser_matches_played', 0),
                'player2_matches_played': row.get('winner_matches_played', 0),
                'player1_win_rate': row.get('loser_win_rate', 0.5),
                'player2_win_rate': row.get('winner_win_rate', 0.5),
                'player1_surface_win_rate': row.get('loser_surface_win_rate', 0.5),
                'player2_surface_win_rate': row.get('winner_surface_win_rate', 0.5),
                'player1_recent_form': row.get('loser_recent_form_10', 0.5),
                'player2_recent_form': row.get('winner_recent_form_10', 0.5),
                'player1_momentum': row.get('loser_momentum_score', 0.5),
                'player2_momentum': row.get('winner_momentum_score', 0.5),
                'player1_days_since': row.get('loser_days_since_last', 30),
                'player2_days_since': row.get('winner_days_since_last', 30),

                # H2H features (same as example1 but from loser's perspective)
                'h2h_matches': total_h2h_matches,
                'h2h_win_rate': loser_h2h_rate,

                # Additional features
                'player1_rank_trend': row.get('loser_rank_trend', 0.0),
                'player2_rank_trend': row.get('winner_rank_trend', 0.0),
                'player1_fatigue': row.get('loser_fatigue', 0.0),
                'player2_fatigue': row.get('winner_fatigue', 0.0),
                'player1_peak_rank': row.get('loser_peak_rank', 2500),
                'player2_peak_rank': row.get('winner_peak_rank', 2500),
                'round_numeric': round_num,

                # Target (0 because player1 lost in this perspective)
                'target': 0
            }

            prediction_data.extend([example1, example2])

            # NOW update H2H stats AFTER creating both examples
            self.h2h_stats[key]['matches'].append(row['tourney_date'])
            if p1_id == key[0]:  # winner is first in sorted key
                self.h2h_stats[key]['p1_wins'] += 1
            else:  # winner is second in sorted key
                self.h2h_stats[key]['p2_wins'] += 1

        pred_df = pd.DataFrame(prediction_data)

        # Create difference features (key for tennis prediction)
        pred_df['rank_diff'] = pred_df['player1_rank'] - \
            pred_df['player2_rank']  # Negative is better for player1
        pred_df['rank_points_diff'] = pred_df['player1_rank_points'] - \
            pred_df['player2_rank_points']
        pred_df['age_diff'] = pred_df['player1_age'] - pred_df['player2_age']
        pred_df['experience_diff'] = pred_df['player1_matches_played'] - \
            pred_df['player2_matches_played']
        pred_df['win_rate_diff'] = pred_df['player1_win_rate'] - \
            pred_df['player2_win_rate']
        pred_df['surface_win_rate_diff'] = pred_df['player1_surface_win_rate'] - \
            pred_df['player2_surface_win_rate']
        pred_df['form_diff'] = pred_df['player1_recent_form'] - \
            pred_df['player2_recent_form']
        pred_df['momentum_diff'] = pred_df['player1_momentum'] - \
            pred_df['player2_momentum']
        pred_df['rest_diff'] = pred_df['player2_days_since'] - \
            pred_df['player1_days_since']  # Positive means player1 more rested

        # Advanced interaction features
        pred_df['rank_experience_interaction'] = pred_df['rank_diff'] * \
            pred_df['experience_diff'] / 1000
        pred_df['form_momentum_combo'] = pred_df['form_diff'] * \
            pred_df['momentum_diff']
        pred_df['surface_specialization'] = pred_df['surface_win_rate_diff'] - \
            pred_df['win_rate_diff']

        # NO LONGER CALL add_h2h_features - we've already added them above
        # pred_df = self.add_h2h_features(pred_df)

        # Encode categorical features
        le_surface = LabelEncoder()
        pred_df['surface_encoded'] = le_surface.fit_transform(
            pred_df['surface'])

        le_level = LabelEncoder()
        pred_df['tourney_level_encoded'] = le_level.fit_transform(
            pred_df['tourney_level'])

        le_round = LabelEncoder()
        pred_df['round_encoded'] = le_round.fit_transform(pred_df['round'])

        pred_df = self.add_advanced_features(pred_df)

        # Select final features for modeling
        self.features = [
            'rank_diff', 'rank_points_diff', 'age_diff', 'experience_diff',
            'win_rate_diff', 'surface_win_rate_diff', 'form_diff', 'momentum_diff',
            'rest_diff', 'rank_experience_interaction', 'form_momentum_combo',
            'surface_specialization', 'surface_encoded', 'tourney_level_encoded',
            'round_encoded', 'best_of', 'draw_size', 'h2h_matches', 'h2h_win_rate',
            'surface_expertise_diff', 'fatigue_diff', 'peak_rank_diff', 'rank_trend_diff',

        ]

        print(
            f"   ✅ Created {len(pred_df)} training examples with {len(self.features)} features")
        print(
            f"   📊 Target distribution: {pred_df['target'].value_counts().to_dict()}")

        return pred_df

    def create_ensemble_model(self, X, y, dates):
        """Train models with hyperparameter tuning"""
        print("🤖 Training prediction models with hyperparameter tuning...")

        # Run hyperparameter optimization
        tuner = TennisHyperparameterTuner(X, y, dates)
        # best_params = tuner.optimize(n_trials=10) hyperparameter tuning
        final_models = {
            "xgb": xgb.XGBClassifier(
                n_estimators=408,
                max_depth=3,
                learning_rate=0.0157,
                subsample=0.7257,
                gamma=0.1893
            ),
            "lgb": lgb.LGBMClassifier(
                n_estimators=499,
                num_leaves=59,
                learning_rate=0.0230,
                feature_fraction=0.8524,
                bagging_fraction=0.8230,
                min_child_samples=47,
                verbose=-1
                ),
            "cat": CatBoostClassifier(
                iterations=731,
                depth=8,
                learning_rate=0.0129,
                l2_leaf_reg=3.995,
                random_strength=0.6598,
                verbose=False,
            )
        }

        # print(f"🎯 Best parameters found:\n{best_params}")

        # Train final model with best parameters
        # final_models = {
        #     'xgb': xgb.XGBClassifier(
        #         n_estimators=best_params['xgb_n_estimators'],
        #         max_depth=best_params['xgb_max_depth'],
        #         learning_rate=best_params['xgb_lr'],
        #         subsample=best_params['xgb_subsample'],
        #         colsample_bytree=best_params['xgb_colsample'],
        #         gamma=best_params['xgb_gamma'],
        #         random_state=42
        #     ),
        #     'lgb': lgb.LGBMClassifier(
        #         n_estimators=best_params['lgb_n_estimators'],
        #         num_leaves=best_params['lgb_num_leaves'],
        #         learning_rate=best_params['lgb_lr'],
        #         feature_fraction=best_params['lgb_feature_fraction'],
        #         bagging_fraction=best_params['lgb_bagging_fraction'],
        #         min_child_samples=best_params['lgb_min_child'],
        #         random_state=42,
        #         verbose=-1
        #     ),
        #     'cat': CatBoostClassifier(
        #         iterations=best_params['cat_iterations'],
        #         depth=best_params['cat_depth'],
        #         learning_rate=best_params['cat_lr'],
        #         l2_leaf_reg=best_params['cat_l2'],
        #         random_strength=best_params['cat_random_strength'],
        #         random_seed=42,
        #         verbose=False
        #     )
        # }

        # Enhanced meta-learner
        meta_learner = LogisticRegression(
            C=0.1,  # More regularization
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        from sklearn.model_selection import StratifiedKFold

        # cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # Time-series aware stacking
        return StackingClassifier(
            estimators=list(final_models.items()),
            final_estimator=meta_learner,
            cv=5,
            passthrough=True,  # Keep original features
            n_jobs=-1  # Parallel processing
        )

    def time_series_cv(self, model, X, y, dates, n_splits=5):
        """Manual time-series validation that works with StackingClassifier"""
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.base import clone

        print("   ⏳ Running time-series cross-validation...")

        # Sort by date
        sorted_idx = dates.argsort()
        X_sorted = X.iloc[sorted_idx]
        y_sorted = y.iloc[sorted_idx]

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sorted)):
            X_train, X_val = X_sorted.iloc[train_idx], X_sorted.iloc[val_idx]
            y_train, y_val = y_sorted.iloc[train_idx], y_sorted.iloc[val_idx]

            # Clone model for this fold
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)

            # Get predictions
            if hasattr(fold_model, 'predict_proba'):
                preds = fold_model.predict_proba(X_val)[:, 1]
            else:
                preds = fold_model.predict(X_val)

            # Calculate AUC
            auc = roc_auc_score(y_val, preds)
            cv_scores.append(auc)
            print(f"     Fold {fold+1} AUC: {auc:.4f}")

        return cv_scores

    def train_models(self, df, use_top_features=12):
        """Train models with proper feature handling
        Args:
            use_top_features: Number of top features to use (None to use all)
        """
        print("🤖 Training prediction models...")

        # 1. Verify features exist in DataFrame
        available_features = set(df.columns)
        valid_features = [f for f in self.features if f in available_features]

        if len(valid_features) != len(self.features):
            missing = set(self.features) - available_features
            print(f"⚠️ Missing features: {missing}")

        # 2. Create feature matrix with only existing features
        X = df[valid_features].copy()
        y = df['target'].copy()
        dates = pd.to_datetime(df['tourney_date'].copy())

        print(f"✅ Using {len(valid_features)} features:")
        print(valid_features)

        # 3. Handle missing values (unchanged)
        print(f"🔍 Checking for missing values...")
        if X.isnull().sum().sum() > 0:
            X = self.advanced_missing_value_handling(X)

        # 4. Time-based split
        split_date = dates.quantile(0.8)
        train_mask = dates <= split_date

        X_train, X_test = X[train_mask], X[~train_mask]
        y_train, y_test = y[train_mask], y[~train_mask]

        print(f"📊 Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # 5. Separate numeric and categorical features
        numeric_features = X_train.select_dtypes(
            include=np.number).columns.tolist()
        print(
            f"🔢 Numeric features ({len(numeric_features)}:", numeric_features)

        # 6. Scale only numeric features
        print("📏 Scaling numeric features...")
        self.scaler.fit(X_train[numeric_features])

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[numeric_features] = self.scaler.transform(
            X_train[numeric_features])
        X_test_scaled[numeric_features] = self.scaler.transform(
            X_test[numeric_features])

        # Train initial model to get feature importance
        print("   🔄 Training initial model for feature selection...")
        initial_model = self.create_ensemble_model(
            X_train_scaled, y_train, dates[train_mask])
        initial_model.fit(X_train_scaled, y_train)

        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': initial_model.named_estimators_['xgb'].feature_importances_
        }).sort_values('importance', ascending=False)

        # Select top features if specified
        if use_top_features and use_top_features < len(valid_features):
            top_features = feature_importance['feature'].head(
                use_top_features).tolist()
            print(f"\n🔝 Selecting top {use_top_features} features:")
            for i, (_, row) in enumerate(feature_importance.head(use_top_features).iterrows()):
                print(
                    f"   {i+1:2d}. {row['feature']:25s} {row['importance']:.4f}")

            # Filter datasets to only top features
            X_train = X_train_scaled[top_features]
            X_test = X_test_scaled[top_features]
            numeric_features = [
                f for f in numeric_features if f in top_features]

            print(
                f"\n📊 New shapes - Train: {X_train.shape}, Test: {X_test.shape}")

        # Train final ensemble model
        print("\n🔄 Training final ensemble model...")
        ensemble_model = self.create_ensemble_model(
            X_train, y_train, dates[train_mask])

        # Time-series CV for Ensemble
        ts_cv_scores_ens = self.time_series_cv(
            ensemble_model, X_train, y_train, dates[train_mask])
        ensemble_model.fit(X_train, y_train)

        ensemble_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]
        ensemble_pred_binary = (ensemble_pred_proba > 0.5).astype(int)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred_binary)
        ensemble_auc = roc_auc_score(y_test, ensemble_pred_proba)

        # Results
        print(f"\n📊 MODEL PERFORMANCE:")
        print(
            f"   Ensemble Time-series CV AUC: {np.mean(ts_cv_scores_ens):.4f} ± {np.std(ts_cv_scores_ens):.4f}")
        print(X_train.columns)
        print(self.features)

        # Print final feature importance
        final_feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': ensemble_model.named_estimators_['xgb'].feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n🎯 FINAL FEATURE IMPORTANCE ORDER:")
        for i, (_, row) in enumerate(final_feature_importance.iterrows()):
            print(f"   {i+1:2d}. {row['feature']:25s} {row['importance']:.4f}")

        # # Cross-validation on training set
        # print(f"\n🔄 Cross-validation on training set...")
        # try:
        #     cv_ens_scores = cross_val_score(ensemble_model, X_train, y_train, cv=5, scoring='accuracy')

        #     print(f"   Ensemble CV Accuracy: {cv_ens_scores.mean():.4f} ± {cv_ens_scores.std():.4f}")
        # except Exception as e:
        #     print(f"   ⚠️  Cross-validation failed: {e}")


        # At the end of your train_models method (replace the current saving code)
        model_pkl_file = "tennisML_model.pkl"

        # Create dictionary with all required components
        save_data = {
            'ensemble_model': ensemble_model,
            'scaler': self.scaler,
            'features': self.features,
            'feature_importance': feature_importance.to_dict()
        }

        # Save using joblib (better for sklearn models)
        joblib.dump(save_data, model_pkl_file)

        print(f"✅ Model and all components saved to {model_pkl_file}")

        return {
            'ensemble_model': ensemble_model,
            'scaler': self.scaler,
            'features': self.features,
            'test_results': {
                'ensemble_accuracy': ensemble_accuracy,
                'ensemble_auc': ensemble_auc
            },
            'feature_importance': feature_importance
        }

    def process_full_pipeline(self, filepath):
        """Run the complete data processing and modeling pipeline"""
        print("🚀 Starting Tennis Prediction Pipeline...")
        start_time = time.time()
        
        # Step 1: Load and clean data
        df = self.load_and_clean_data(filepath)
        
        # Step 2: Calculate historical stats (this prevents data leakage)
        df_with_history = self.calculate_historical_stats(df)
        
        # Step 3: Create prediction features
        prediction_df = self.create_prediction_features(df_with_history)
        
        # Step 4: Train models
        results = self.train_models(prediction_df)
        
        total_time = time.time() - start_time
        print(f"\n✅ Pipeline completed in {total_time:.2f} seconds!")
        
        return results, prediction_df

# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = TennisDataProcessor()

    # Run pipeline (replace with your file path)
    results, processed_data = processor.process_full_pipeline('tennis_atp_2000_plus_matches.csv')
    
