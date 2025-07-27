"""This is cheating somehow"""


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import joblib
from tqdm.auto import tqdm
import time
import warnings
from xgboost import XGBClassifier
from sklearn import set_config
from sklearn.utils.metadata_routing import MetadataRouter, MetadataRequest
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Enable metadata routing globally at the start of your script
set_config(enable_metadata_routing=True)
# Configure warnings
warnings.filterwarnings("ignore", category=UserWarning)

class EnhancedTennisPredictor:
    class _EarlyStopXGB(BaseEstimator, ClassifierMixin):
        """XGBoost wrapper that implements early stopping"""
        def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.05,
                    early_stopping_rounds=20, random_state=42):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.learning_rate = learning_rate
            self.early_stopping_rounds = early_stopping_rounds
            self.random_state = random_state
            self.model_ = None

        def fit(self, X, y, eval_set=None):
            X, y = check_X_y(X, y)
            self.classes_ = np.unique(y)
            
            params = {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'early_stopping_rounds': self.early_stopping_rounds,
                'random_state': self.random_state,
                'eval_metric': 'logloss',
                'tree_method': 'hist'
            }
            
            self.model_ = XGBClassifier(**params)
            
            if eval_set:
                self.model_.fit(X, y, eval_set=eval_set, verbose=False)
            else:
                self.model_.fit(X, y)
            return self
            
        def predict(self, X):
            check_is_fitted(self)
            X = check_array(X)
            return self.model_.predict(X)
            
        def predict_proba(self, X):
            check_is_fitted(self)
            X = check_array(X)
            return self.model_.predict_proba(X)
            
        def get_params(self, deep=True):
            return {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'early_stopping_rounds': self.early_stopping_rounds,
                'random_state': self.random_state
            }
            
        def set_params(self, **params):
            for param, value in params.items():
                setattr(self, param, value)
            return self
        
        @property
        def feature_importances_(self):
            """Expose the underlying XGBoost feature importances"""
            check_is_fitted(self)
            return self.model_.feature_importances_

    def __init__(self):
        self.features = None
        self.model = None
        self.preprocessor = make_pipeline(
            SimpleImputer(strategy='median'),
        )

    def load_data(self, filepath):
        """Optimized data loading with chunking for large files"""
        print("\n[1/4] Loading data...")
        start = time.time()
        
        try:
            chunks = pd.read_csv(filepath, chunksize=50000)
            df = pd.concat(chunks, ignore_index=True)
            
            required_cols = [
                'tourney_date', 'winner_id', 'loser_id', 'surface',
                'winner_rank_points', 'loser_rank_points',
                'winner_clay_exp', 'loser_clay_exp',
                'winner_recent_wins', 'loser_recent_wins',
                'winner_age', 'loser_age'
            ]
            
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
                
            df['tourney_date'] = pd.to_datetime(df['tourney_date'])
            print(f"✓ Loaded {len(df)} matches in {time.time()-start:.2f}s")
            return df.dropna(subset=['tourney_date'])
            
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {str(e)}")


    def create_balanced_dataset(self, df):
        """Optimized dataset balancing"""
        print("\n[2/4] Creating balanced dataset...")
        start = time.time()
        
        winner_data = df.copy()
        loser_data = df.copy()
        
        swap_cols = ['id', 'rank_points', 'clay_exp', 'recent_wins', 'age']
        for col in swap_cols:
            if f'winner_{col}' in df.columns:
                winner_col = f'winner_{col}'
                loser_col = f'loser_{col}'
                loser_data[[winner_col, loser_col]] = loser_data[[loser_col, winner_col]].values
        
        winner_data['target'] = 1
        loser_data['target'] = 0
        balanced_df = pd.concat([winner_data, loser_data], ignore_index=True)
        
        print(f"✓ Created {len(balanced_df)} records in {time.time()-start:.2f}s")
        return balanced_df
    def prepare_features(self, df):
        """Enhanced feature engineering - DATA LEAKAGE FREE VERSION"""
        print("\n[3/4] Preparing features...")
        start = time.time()
        
        # DEBUG: Check for data leakage in column names
        print(f"DEBUG: Available columns: {list(df.columns)}")
        leakage_columns = [col for col in df.columns if 'winner_' in col or 'loser_' in col]
        if leakage_columns:
            print(f"⚠️  WARNING: Potential data leakage columns detected: {leakage_columns}")
        
        # ASSUMPTION: Your data should have columns like:
        # player1_rank_points, player2_rank_points, player1_clay_exp, player2_clay_exp, etc.
        # If not, you need to restructure your data first!
        
        required_cols = ['player1_rank_points', 'player2_rank_points', 'player1_clay_exp', 
                        'player2_clay_exp', 'player1_recent_wins', 'player2_recent_wins',
                        'player1_age', 'player2_age', 'player1_id', 'player2_id']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ ERROR: Missing required columns: {missing_cols}")
            print("You need to restructure your data to have player1/player2 format instead of winner/loser")
            return df
        
        print("✓ Data structure validation passed")
        
        # CORRECTED FEATURES - No data leakage
        df['rank_diff'] = df['player1_rank_points'] - df['player2_rank_points']
        df['exp_diff'] = df['player1_clay_exp'] - df['player2_clay_exp']  
        df['recent_wins_diff'] = df['player1_recent_wins'] - df['player2_recent_wins']
        df['age_diff'] = df['player1_age'] - df['player2_age']
        
        # Surface feature (if available)
        if 'surface' in df.columns:
            df['is_clay'] = df['surface'].eq('Clay').astype(int)
            print(f"DEBUG: Clay matches: {df['is_clay'].sum()}/{len(df)}")
        
        # Advanced features
        df['momentum'] = df['recent_wins_diff'] * df['exp_diff']
        df['rank_ratio'] = np.where(
            df['player2_rank_points'] == 0,
            1,
            df['player1_rank_points'] / df['player2_rank_points']
        )
        df['experience_gap'] = np.log1p(df['player1_clay_exp']) - np.log1p(df['player2_clay_exp'])
        
        # Days since last match (temporal features)
        if 'tourney_date' in df.columns:
            df['player1_days_since'] = df.groupby('player1_id')['tourney_date'].diff().dt.days.fillna(0)
            df['player2_days_since'] = df.groupby('player2_id')['tourney_date'].diff().dt.days.fillna(0)
            df['days_since_diff'] = df['player1_days_since'] - df['player2_days_since']
            
            # DEBUG: Check temporal features
            print(f"DEBUG: Player1 avg days since last match: {df['player1_days_since'].mean():.1f}")
            print(f"DEBUG: Player2 avg days since last match: {df['player2_days_since'].mean():.1f}")
        
        df['relative_momentum'] = df['recent_wins_diff'] / (df['age_diff'].abs() + 1)
        df['rank_pressure'] = np.where(
            df['player1_rank_points'] > df['player2_rank_points'],
            df['player1_rank_points'] - df['player2_rank_points'],
            0
        )
        
        # NEW ADVANCED FEATURES for better performance
        
        # 1. Head-to-head record (if available)
        if 'h2h_player1_wins' in df.columns and 'h2h_total_matches' in df.columns:
            df['h2h_win_rate'] = df['h2h_player1_wins'] / (df['h2h_total_matches'] + 1)  # +1 to avoid division by zero
            df['h2h_experience'] = np.log1p(df['h2h_total_matches'])
            print(f"DEBUG: Head-to-head data available for {(df['h2h_total_matches'] > 0).sum()} matches")
        
        # 2. Ranking momentum (change in rank over time)
        if 'player1_rank_change_30d' in df.columns:
            df['rank_momentum_diff'] = df['player1_rank_change_30d'] - df['player2_rank_change_30d']
            df['combined_momentum'] = df['rank_momentum_diff'] * df['recent_wins_diff']
        
        # 3. Form-based features
        df['rank_exp_interaction'] = df['rank_diff'] * df['exp_diff'] / 1000  # Normalized
        df['age_exp_balance'] = (df['player1_age'] * df['player1_clay_exp']) - (df['player2_age'] * df['player2_clay_exp'])
        
        # 4. Fatigue/Rest indicators
        if 'tourney_date' in df.columns:
            df['rest_advantage'] = np.where(
                df['days_since_diff'] > 0,
                np.log1p(df['days_since_diff']),
                -np.log1p(abs(df['days_since_diff']))
            )
        
        # 5. Tournament-specific features
        if 'tourney_level' in df.columns:
            # Encode tournament importance
            tourney_importance = {'G': 4, 'M': 3, 'A': 2, 'C': 1}  # Grand Slam, Masters, ATP, Challenger
            df['tourney_importance'] = df['tourney_level'].map(tourney_importance).fillna(1)
            df['pressure_factor'] = df['rank_diff'] * df['tourney_importance']
        
        # 6. Physical/endurance features
        if 'avg_match_duration' in df.columns:
            df['endurance_diff'] = df['player1_avg_match_duration'] - df['player2_avg_match_duration'] 
            df['endurance_age_factor'] = df['endurance_diff'] * df['age_diff']
        
        # Feature list - NO LEAKAGE
        base_features = [
            'rank_diff', 'exp_diff', 'recent_wins_diff', 'age_diff', 
            'momentum', 'rank_ratio', 'experience_gap', 'relative_momentum', 
            'rank_pressure', 'rank_exp_interaction', 'age_exp_balance'
        ]
        
        # Add conditional features
        if 'surface' in df.columns:
            base_features.append('is_clay')
        
        if 'tourney_date' in df.columns:
            base_features.extend(['player1_days_since', 'player2_days_since', 'days_since_diff', 'rest_advantage'])
        
        if 'h2h_player1_wins' in df.columns:
            base_features.extend(['h2h_win_rate', 'h2h_experience'])
        
        if 'player1_rank_change_30d' in df.columns:
            base_features.extend(['rank_momentum_diff', 'combined_momentum'])
        
        if 'tourney_level' in df.columns:
            base_features.extend(['tourney_importance', 'pressure_factor'])
        
        if 'avg_match_duration' in df.columns:
            base_features.extend(['endurance_diff', 'endurance_age_factor'])
        
        self.features = base_features
        
        # COMPREHENSIVE DEBUG INFORMATION
        print(f"\n🔍 FEATURE ENGINEERING DEBUG:")
        print(f"   • Total features created: {len(self.features)}")
        print(f"   • Feature list: {self.features}")
        
        # Check for any remaining data leakage
        for feature in self.features:
            if feature in df.columns:
                if df[feature].isnull().sum() > 0:
                    print(f"   ⚠️  {feature}: {df[feature].isnull().sum()} missing values")
                
                # Statistical sanity checks
                if 'diff' in feature:
                    print(f"   📊 {feature}: mean={df[feature].mean():.3f}, std={df[feature].std():.3f}")
        
        # Final data leakage check
        correlation_with_target = {}
        if 'target' in df.columns or 'player1_won' in df.columns:
            target_col = 'target' if 'target' in df.columns else 'player1_won'
            for feature in self.features:
                if feature in df.columns:
                    corr = df[feature].corr(df[target_col])
                    correlation_with_target[feature] = corr
                    if abs(corr) > 0.8:  # Suspiciously high correlation
                        print(f"   🚨 HIGH CORRELATION WARNING: {feature} has correlation {corr:.3f} with target")
        
        print(f"✅ Feature engineering completed in {time.time()-start:.2f}s")
        
        # Additional validation
        print(f"\n📈 FEATURE STATISTICS:")
        for feature in self.features[:5]:  # Show stats for first 5 features
            if feature in df.columns:
                print(f"   {feature}: min={df[feature].min():.3f}, max={df[feature].max():.3f}, mean={df[feature].mean():.3f}")
        
        return df

    # ADDITIONAL HELPER FUNCTION: Data structure converter
    def convert_winner_loser_to_player_format(df):
        """Convert winner/loser format to player1/player2 format to prevent data leakage"""
        print("Converting data from winner/loser format to player1/player2 format...")
        
        # Create two rows for each match (player1 perspective and player2 perspective)
        matches_p1 = df.copy()
        matches_p1['player1_id'] = df['winner_id']
        matches_p1['player2_id'] = df['loser_id'] 
        matches_p1['player1_rank_points'] = df['winner_rank_points']
        matches_p1['player2_rank_points'] = df['loser_rank_points']
        matches_p1['player1_clay_exp'] = df['winner_clay_exp']
        matches_p1['player2_clay_exp'] = df['loser_clay_exp']
        matches_p1['player1_recent_wins'] = df['winner_recent_wins']
        matches_p1['player2_recent_wins'] = df['loser_recent_wins']
        matches_p1['player1_age'] = df['winner_age']
        matches_p1['player2_age'] = df['loser_age']
        matches_p1['player1_won'] = 1  # Target variable
        
        matches_p2 = df.copy()
        matches_p2['player1_id'] = df['loser_id']
        matches_p2['player2_id'] = df['winner_id']
        matches_p2['player1_rank_points'] = df['loser_rank_points']
        matches_p2['player2_rank_points'] = df['winner_rank_points']
        matches_p2['player1_clay_exp'] = df['loser_clay_exp']
        matches_p2['player2_clay_exp'] = df['winner_clay_exp']
        matches_p2['player1_recent_wins'] = df['loser_recent_wins']
        matches_p2['player2_recent_wins'] = df['winner_recent_wins']
        matches_p2['player1_age'] = df['loser_age']
        matches_p2['player2_age'] = df['winner_age']
        matches_p2['player1_won'] = 0  # Target variable
        
        # Combine both perspectives
        balanced_df = pd.concat([matches_p1, matches_p2], ignore_index=True)
        
        # Drop original winner/loser columns to prevent accidental usage
        cols_to_drop = [col for col in balanced_df.columns if 'winner_' in col or 'loser_' in col]
        balanced_df = balanced_df.drop(columns=cols_to_drop)
        
        print(f"✅ Converted {len(df)} matches to {len(balanced_df)} training examples")
        return balanced_df
    # def prepare_features(self, df):
    #     """Enhanced feature engineering"""
    #     print("\n[3/4] Preparing features...")
    #     start = time.time()
        
    #     df['rank_diff'] = df['winner_rank_points'] - df['loser_rank_points']
    #     df['exp_diff'] = df['winner_clay_exp'] - df['loser_clay_exp']
    #     df['recent_wins_diff'] = df['winner_recent_wins'] - df['loser_recent_wins']
    #     df['age_diff'] = df['winner_age'] - df['loser_age']
    #     # df['is_clay'] = df['surface'].eq('Clay').astype(int)
    #     df['momentum'] = df['recent_wins_diff'] * df['exp_diff']
    #     df['rank_ratio'] = np.where(
    #         df['loser_rank_points'] == 0, 
    #         1, 
    #         df['winner_rank_points'] / df['loser_rank_points']
    #     )
    #     df['experience_gap'] = np.log1p(df['winner_clay_exp']) - np.log1p(df['loser_clay_exp'])
    #     # df['days_since_last_match'] = df.groupby('winner_id')['tourney_date'].diff().dt.days.fillna(0)
    #     df['winner_days_since'] = df.groupby('winner_id')['tourney_date'].diff().dt.days.fillna(0)
    #     df['loser_days_since'] = df.groupby('loser_id')['tourney_date'].diff().dt.days.fillna(0)
    #     df['days_since_last_match'] = df['winner_days_since'] - df['loser_days_since']

    #     df['relative_momentum'] = df['recent_wins_diff'] / (df['age_diff'].abs() + 1)
    #     df['rank_pressure'] = np.where(
    #         df['winner_rank_points'] > df['loser_rank_points'],
    #         df['winner_rank_points'] - df['loser_rank_points'],
    #         0
    #     )
    #     df['recovery_indicator'] = df['days_since_last_match'] * df['recent_wins_diff']

    #     self.features = [
    #         'rank_diff', 'exp_diff', 'recent_wins_diff', 
    #         'age_diff', 'momentum', 'rank_ratio',
    #         'experience_gap', 'days_since_last_match',
    #         'winner_days_since', 'loser_days_since',
    #         'relative_momentum', 'rank_pressure',
    #     ]
        
    #     print(f"✓ Prepared {len(self.features)} features in {time.time()-start:.2f}s")
    #     return df
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train the stacked model with early stopping"""
        print("\n[4/4] Training optimized stacked model...")
        start = time.time()

        # Pre-train XGBoost to determine optimal number of trees
        print("Pre-training XGBoost with early stopping...")
        xgb_pre = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            early_stopping_rounds=20,
            random_state=42,
            eval_metric='logloss'
        )
        
        xgb_pre.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        best_n_estimators = xgb_pre.best_iteration or 100

        # Define base models
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            max_features=0.8,
            min_samples_leaf=2,
            min_samples_split=10,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )

        xgb = self._EarlyStopXGB(
            n_estimators=best_n_estimators,
            max_depth=6,
            learning_rate=0.05,
            early_stopping_rounds=None,  # Already determined during pre-training
            random_state=42
        )

        # Stacking classifier
        stack = StackingClassifier(
            estimators=[('rf', rf), ('xgb', xgb)],
            final_estimator=LogisticRegression(
                penalty='l2',
                C=0.1,
                max_iter=1000,
                n_jobs=-1,
                random_state=42
            ),
            passthrough=False,
            n_jobs=-1,
            cv=3
        )

        # Train final model
        with tqdm(total=100, desc="Training Stacked Model") as pbar:
            stack.fit(X_train, y_train)
            pbar.update(100)

        self.model = stack
        train_time = time.time() - start
        
        self._evaluate_model(X_train, y_train, X_val, y_val)
        print(f"\n✓ Optimized stacked model trained in {train_time:.2f}s")
        print(f"XGBoost used {best_n_estimators} trees (early stopping)")
        return self.model
    
    def _evaluate_model(self, X_train, y_train, X_test, y_test):
        """Enhanced model evaluation with feature importance handling"""
        print("\n=== Model Evaluation ===")
        
        results = []
        for X, y, name in [(X_train, y_train, "Training"), (X_test, y_test, "Validation")]:
            preds = self.model.predict(X)
            probs = self.model.predict_proba(X)[:, 1]
            
            metrics = {
                'Dataset': name,
                'Accuracy': accuracy_score(y, preds),
                'ROC AUC': roc_auc_score(y, probs),
                'Precision': classification_report(y, preds, output_dict=True)['weighted avg']['precision'],
                'Recall': classification_report(y, preds, output_dict=True)['weighted avg']['recall'],
                'F1': classification_report(y, preds, output_dict=True)['weighted avg']['f1-score']
            }
            results.append(metrics)
        
        print(pd.DataFrame(results).to_markdown(tablefmt="grid"))
        
        # Get feature importances from base models
        print("\n🔍 Feature Importances:")
        
        try:
            # Get importances from RandomForest
            rf_imp = self.model.estimators_[0].feature_importances_
            
            # Get importances from our XGBoost wrapper
            xgb_imp = self.model.estimators_[1].feature_importances_
            
            imp_df = pd.DataFrame({
                'Feature': self.features,
                'RF Importance': rf_imp,
                'XGB Importance': xgb_imp
            }).sort_values('RF Importance', ascending=False)
            
            print(imp_df.to_markdown(tablefmt="grid"))
        except Exception as e:
            print(f"Could not display feature importances: {str(e)}")


    def _make_player_temporal_split(self, df, val_ratio=0.2, test_ratio=0.2,
                                    min_val_size=500, min_test_size=1000,
                                    date_col='tourney_date', id_cols=('winner_id', 'loser_id')):
        """
        Creates a player-exclusive temporal split:
            - All matches sorted by date
            - Players in val/test are excluded from train
            - Ensures no temporal leakage
            - Enforces minimum sizes for val/test sets
        """
        if df.empty:
            return None, None, None

        winner_id, loser_id = id_cols

        # Auto-detect date column if not provided
        if date_col not in df.columns:
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if not date_cols:
                raise ValueError("No valid date column found for splitting")
            date_col = date_cols[0]

        print(f"[INFO] Using date column: {date_col}")
        print(f"[INFO] Excluding players in val/test from train")

        # Sort by date and reset index
        df = df.sort_values(date_col).reset_index(drop=True)

        # Identify total size
        total_size = len(df)
        test_start = int(total_size * (1 - test_ratio))
        val_start = int(test_start * (1 - val_ratio))

        # Initial val/test sets used to identify exclusive players
        test_initial = df.iloc[test_start:]
        val_initial = df.iloc[val_start:test_start]

        # Get all players in val/test
        val_test_players = set(val_initial[winner_id]).union(val_initial[loser_id]) \
                        .union(set(test_initial[winner_id])).union(test_initial[loser_id])

        # Filter train to exclude any match involving val/test players
        train_mask = (~df[winner_id].isin(val_test_players)) & \
                    (~df[loser_id].isin(val_test_players)) & \
                    (df[date_col] < val_initial[date_col].min())

        train = df[train_mask]
        
        # Rebuild val and test using original cutoffs
        val = df[(df.index >= val_start) & (df.index < test_start)]
        test = df[df.index >= test_start]

        # Enforce minimum sizes
        if len(val) < min_val_size or len(test) < min_test_size or len(train) == 0:
            print("✗ Could not create valid split with exclusive players.")
            return None, None, None

        print(f"✓ Final Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")

        return train, val, test
    
    def save_splits_to_csv(self, train, val, test, output_dir="data/splits"):
        os.makedirs(output_dir, exist_ok=True)
        train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        val.to_csv(os.path.join(output_dir, "val.csv"), index=False)
        test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
        print(f"\n✓ Splits saved to: {os.path.abspath(output_dir)}")

    def log_overlapping_players(self, train, val, test, winner_id='winner_id', loser_id='loser_id'):
        def get_players(df): return set(df[winner_id]).union(df[loser_id])
        train_players = get_players(train)
        val_players = get_players(val)
        test_players = get_players(test)
        overlap_val = train_players & val_players
        overlap_test = train_players & test_players
        print("\n[!] Running integrity checks...")
        print("✓ No overlapping players between train and validation sets" if not overlap_val
              else f"⚠ Warning: {len(overlap_val)} overlapping players between train and validation set:\n{list(overlap_val)[:5]}...")
        print("✓ No overlapping players between train and test sets" if not overlap_test
              else f"⚠ Warning: {len(overlap_test)} overlapping players between train and test set:\n{list(overlap_test)[:5]}...")
        print(f"✓ Unique players in train: {len(train_players)}")

    def plot_player_timeline(self, train, val, test, date_col='tourney_date', id_cols=('winner_id', 'loser_id')):
        winner_id, loser_id = id_cols
        def extract_player_dates(df):
            df = df[[date_col, winner_id, loser_id]]
            df_wide = pd.melt(df, id_vars=[date_col], value_vars=[winner_id, loser_id],
                            var_name='role', value_name='player_id')
            return df_wide.groupby('player_id')[date_col].min().reset_index()
        train_players = extract_player_dates(train)
        val_players = extract_player_dates(val)
        test_players = extract_player_dates(test)
        train_players['set'] = 'Train'
        val_players['set'] = 'Validation'
        test_players['set'] = 'Test'
        all_players = pd.concat([train_players, val_players, test_players], ignore_index=True)
        plt.figure(figsize=(12, 6))
        sns.stripplot(x=date_col, y='set', hue='set', data=all_players, jitter=False, palette='Set2', size=4)
        plt.title("Player First Appearance Timeline Across Train/Val/Test Sets")
        plt.xlabel("First Match Date")
        plt.ylabel("Dataset Split")
        plt.legend([], [], frameon=False)
        plt.tight_layout()
        plt.savefig("data/splits/player_timeline.png")
        plt.show()
        print("✓ Player timeline visualization saved to: data/splits/player_timeline.png")

def main():
    try:
        predictor = EnhancedTennisPredictor()
        # Load and balance data
        df = predictor.load_data('clay_matches.csv')
        balanced_df = predictor.create_balanced_dataset(df)
        processed_df = predictor.prepare_features(balanced_df)

        print("\n[4/4] Creating player-exclusive temporal splits...")
        train, val, test = predictor._make_player_temporal_split(
            processed_df,
            val_ratio=0.1,
            test_ratio=0.2,
            min_val_size=500,
            min_test_size=3000,
            date_col='tourney_date'
        )
        if any(x is None for x in [train, val, test]):
            print("✗ Could not create a valid temporal split with exclusive players.")
            return
        print(f"✓ Train records: {len(train)}")
        print(f"✓ Validation records: {len(val)}")
        print(f"✓ Test records: {len(test)}")
        print(f"✓ Train date range: {train['tourney_date'].min()} to {train['tourney_date'].max()}")
        print(f"✓ Validation date range: {val['tourney_date'].min()} to {val['tourney_date'].max()}")
        print(f"✓ Test date range: {test['tourney_date'].min()} to {test['tourney_date'].max()}")
        # Save and validate
        predictor.save_splits_to_csv(train, val, test)
        predictor.log_overlapping_players(train, val, test)
        
        # Preprocess
        X_train = predictor.preprocessor.fit_transform(train[predictor.features])
        y_train = train['target']
        X_val = predictor.preprocessor.transform(val[predictor.features])
        y_val = val['target']
        X_test = predictor.preprocessor.transform(test[predictor.features])
        y_test = test['target']

        # Run integrity checks
        print("\n[!] Running integrity checks...")
        def get_players(df): return set(df['winner_id']).union(df['loser_id'])
        train_players = get_players(train)
        val_players = get_players(val)
        test_players = get_players(test)
        overlap_val = train_players & val_players
        overlap_test = train_players & test_players
        print("✓ No overlapping players between train and validation sets" if not overlap_val
              else f"⚠ Warning: {len(overlap_val)} overlapping players between train and validation set")
        print("✓ No overlapping players between train and test sets" if not overlap_test
              else f"⚠ Warning: {len(overlap_test)} overlapping players between train and test set")
        print(f"✓ Unique players in train: {len(train_players)}")

        # Print class balance
        def print_class_balance(name, target):
            balance = target.value_counts(normalize=True).to_dict()
            print(f"{name} set balance: {balance}")
        print_class_balance("Train", y_train)
        print_class_balance("Validation", y_val)
        print_class_balance("Test", y_test)

        # Plot player timelines
        predictor.plot_player_timeline(train, val, test)

        # Train model
        model = predictor.train_model(X_train, y_train, X_val, y_val)

        print("\n=== Final Test Evaluation ===")
        test_preds = model.predict(X_test)
        test_probs = model.predict_proba(X_test)[:, 1]
        print(f"Test ROC AUC: {roc_auc_score(y_test, test_probs):.4f}")
        print(classification_report(y_test, test_preds))

        joblib.dump(model, 'stacked_tennis_model.pkl')
        print("\n✔ Model saved to stacked_tennis_model.pkl")

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")


if __name__ == "__main__":
    main()