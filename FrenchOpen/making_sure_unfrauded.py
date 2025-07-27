"""
Enhanced Tennis Match Prediction Model (No Cheating Allowed)
"""

from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report,log_loss
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from xgboost import XGBClassifier
import joblib
from tqdm.auto import tqdm
import time
import warnings
import os
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score, log_loss, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

class EnhancedTennisPredictor:
    class _EarlyStopXGB(BaseEstimator, ClassifierMixin):
        """XGBoost wrapper that implements early stopping without leakage."""
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
            check_is_fitted(self)
            return self.model_.feature_importances_

    def __init__(self):
        self.model = None
        self.features = []
        self.preprocessor = make_pipeline(SimpleImputer(strategy='median'))

import pandas as pd
import numpy as np
import time
import os
import joblib
from datetime import datetime, timedelta
from collections import defaultdict, deque
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, log_loss, classification_report, confusion_matrix
)
from xgboost import XGBClassifier

class LeakageDetector:
    """Utility class to detect and prevent data leakage"""
    
    @staticmethod
    def check_temporal_leakage(train_dates, val_dates, test_dates):
        """Ensure no future information leaks into past splits"""
        print("\n🔍 Checking Temporal Leakage...")
        
        train_max = train_dates.max() if len(train_dates) > 0 else pd.Timestamp.min
        val_min = val_dates.min() if len(val_dates) > 0 else pd.Timestamp.max
        val_max = val_dates.max() if len(val_dates) > 0 else pd.Timestamp.min
        test_min = test_dates.min() if len(test_dates) > 0 else pd.Timestamp.max
        
        temporal_ok = (train_max < val_min) and (val_max < test_min)
        
        if temporal_ok:
            print("✅ No temporal leakage detected")
        else:
            print("❌ TEMPORAL LEAKAGE DETECTED!")
            print(f"Train max date: {train_max}")
            print(f"Val min date: {val_min}")
            print(f"Test min date: {test_min}")
            
        return temporal_ok
    
    @staticmethod
    def check_player_leakage(train_players, val_players, test_players):
        """Check for player overlap between splits"""
        print("\n🔍 Checking Player Leakage...")
        
        train_val_overlap = len(train_players & val_players)
        train_test_overlap = len(train_players & test_players)
        val_test_overlap = len(val_players & test_players)
        
        print(f"Train-Val player overlap: {train_val_overlap}")
        print(f"Train-Test player overlap: {train_test_overlap}")
        print(f"Val-Test player overlap: {val_test_overlap}")
        
        if train_val_overlap == 0 and train_test_overlap == 0:
            print("✅ No player leakage detected")
            return True
        else:
            print("❌ PLAYER LEAKAGE DETECTED!")
            return False
    
    @staticmethod
    def check_duplicate_matches(df, match_cols=['winner_id', 'loser_id', 'tourney_date']):
        """Check for duplicate matches that could cause leakage"""
        print("\n🔍 Checking for Duplicate Matches...")
        
        # Check exact duplicates
        exact_dupes = df.duplicated(subset=match_cols).sum()
        
        # Check reverse duplicates (A beats B vs B beats A on same date)
        df_reverse = df.copy()
        df_reverse[['winner_id', 'loser_id']] = df_reverse[['loser_id', 'winner_id']].values
        reverse_dupes = pd.merge(df, df_reverse, on=match_cols, how='inner').shape[0]
        
        print(f"Exact duplicate matches: {exact_dupes}")
        print(f"Reverse duplicate matches: {reverse_dupes}")
        
        if exact_dupes > 0 or reverse_dupes > 0:
            print("❌ DUPLICATE MATCHES DETECTED!")
            return False
        else:
            print("✅ No duplicate matches detected")
            return True

class EnhancedTennisPredictor:
    def __init__(self):
        self.model = None
        self.features = []
        self.feature_names = []
        self.preprocessor = None
        self.leakage_detector = LeakageDetector()
        
        # Enhanced preprocessing pipeline
        self.setup_preprocessor()
        
    def update_preprocessor(self, selected_features, feature_df=None):
        """Update preprocessor with only features that exist in the data"""
        # Use the feature dataframe if provided, otherwise fall back to self.df
        reference_df = feature_df if feature_df is not None else self.df
        
        print(f"🔍 DEBUG - update_preprocessor:")
        print(f"Selected features: {len(selected_features)}")
        print(f"Reference df columns: {len(reference_df.columns)}")
        print(f"Reference df shape: {reference_df.shape}")
        
        # Verify features exist in the reference data
        existing_features = [f for f in selected_features if f in reference_df.columns]
    
        if len(existing_features) < len(selected_features):
            missing = set(selected_features) - set(existing_features)
            print(f"⚠️ Warning: Dropping missing features: {missing}")
            print(f"Available columns in reference_df: {list(reference_df.columns)}")
        else:
            print("✅ All selected features found in reference dataframe")
    
        if not existing_features:
            print("❌ CRITICAL: No features exist in reference dataframe!")
            print(f"Selected features: {selected_features}")
            print(f"Available columns: {list(reference_df.columns)}")
            raise ValueError("No valid features found for preprocessing")
    
        # Create numeric transformer pipeline
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
    
        # Update the preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, existing_features)
            ],
            remainder='drop',
            verbose_feature_names_out=False
        )
    
        # Store the validated features
        self.features = existing_features
        print(f"✅ Preprocessor updated with {len(existing_features)} valid features")
        return existing_features

    def setup_preprocessor(self):
        """Setup robust preprocessing pipeline"""
        self.numeric_features = []
        self.preprocessor = ColumnTransformer(
            transformers=[],  # Will be populated after feature selection
            remainder='drop'  # Drop any columns not explicitly processed
        )
        
        print("✅ Preprocessor setup complete - will be populated after feature selection")

    def preprocess_features(self, df, fit=True):
        """Preprocess features with proper validation"""
        print(f"🔍 DEBUG - preprocess_features:")
        print(f"Input df shape: {df.shape}")
        print(f"Features to use: {len(self.features)}")
        print(f"Preprocessor configured: {self.preprocessor is not None}")
        
        # Ensure we have the right features
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            print(f"❌ Missing features in input df: {missing_features}")
            available_features = [f for f in self.features if f in df.columns]
            if not available_features:
                raise ValueError("No features available for preprocessing")
            print(f"⚠️ Using only available features: {len(available_features)}")
            self.features = available_features
            # Update preprocessor with available features
            self.update_preprocessor(self.features, df)
        
        # Select only the features we need
        feature_df = df[self.features].copy()
        print(f"Feature selection shape: {feature_df.shape}")
        
        # Apply preprocessing
        if fit:
            print("🔧 Fitting and transforming features...")
            processed = self.preprocessor.fit_transform(feature_df)
        else:
            print("🔧 Transforming features...")
            processed = self.preprocessor.transform(feature_df)
        
        print(f"Processed shape: {processed.shape}")
        
        if processed.shape[1] == 0:
            print("❌ WARNING: Preprocessing returned empty feature matrix!")
            print(f"Input features: {self.features}")
            print(f"Preprocessor transformers: {self.preprocessor.transformers}")
            
        return processed

    def load_data(self, filepath):
        """Load and validate data with leakage checks"""
        print("[1/5] Loading and validating data...")
        start = time.time()
        
        try:
            df = pd.read_csv(filepath)
            
            # Required columns check
            required_cols = [
                'tourney_date', 'winner_id', 'loser_id', 
                'winner_rank_points', 'loser_rank_points',
                'winner_clay_exp', 'loser_clay_exp',
                'winner_recent_wins', 'loser_recent_wins',
                'winner_age', 'loser_age'
            ]
            
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            
            # Convert dates and validate
            df['tourney_date'] = pd.to_datetime(df['tourney_date'])
            df = df.dropna(subset=['tourney_date']).reset_index(drop=True)
            
            # Check for data quality issues
            self.leakage_detector.check_duplicate_matches(df)
            
            # Remove any exact duplicates
            initial_len = len(df)
            df = df.drop_duplicates(subset=['winner_id', 'loser_id', 'tourney_date'])
            if len(df) < initial_len:
                print(f"⚠️ Removed {initial_len - len(df)} duplicate matches")
            
            # Sort by date for temporal consistency
            df = df.sort_values('tourney_date').reset_index(drop=True)
            
            print(f"✅ Loaded {len(df)} clean matches in {time.time()-start:.2f}s")
            print(f"Date range: {df['tourney_date'].min()} to {df['tourney_date'].max()}")
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {str(e)}")

    def create_match_features(self, df):
        """Create match-level features with proper target balancing"""
        print("\n[2/5] Creating match-level features...")
        start = time.time()
        
        # Create both positive and negative examples
        winner_df = df.copy()
        loser_df = df.copy()
        
        # Positive examples (winner wins)
        winner_df['target'] = 1
        
        # Negative examples (loser wins) - we'll swap features
        loser_df['target'] = 0
        
        # For negative examples, swap winner/loser features
        swap_cols = {
            'winner_id': 'loser_id',
            'loser_id': 'winner_id',
            'winner_rank_points': 'loser_rank_points',
            'loser_rank_points': 'winner_rank_points',
            'winner_age': 'loser_age',
            'loser_age': 'winner_age',
            'winner_clay_exp': 'loser_clay_exp',
            'loser_clay_exp': 'winner_clay_exp',
            'winner_recent_wins': 'loser_recent_wins',
            'loser_recent_wins': 'winner_recent_wins'
        }
        
        loser_df = loser_df.rename(columns=swap_cols)
        
        # Combine both sets
        enhanced_df = pd.concat([winner_df, loser_df], ignore_index=True)
        
        # Shuffle the dataset
        enhanced_df = enhanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Feature calculations (now works for both positive and negative examples)
        enhanced_df['rank_diff'] = enhanced_df['winner_rank_points'] - enhanced_df['loser_rank_points']
        enhanced_df['rank_ratio'] = np.log1p(enhanced_df['winner_rank_points']) / (np.log1p(enhanced_df['loser_rank_points']) + 1e-8)
        enhanced_df['age_diff'] = enhanced_df['winner_age'] - enhanced_df['loser_age']
        enhanced_df['exp_diff'] = np.log1p(enhanced_df['winner_clay_exp']) - np.log1p(enhanced_df['loser_clay_exp'])
        enhanced_df['form_diff'] = enhanced_df['winner_recent_wins'] - enhanced_df['loser_recent_wins']
        
        print(f"✅ Created {len(enhanced_df)} balanced match records in {time.time()-start:.2f}s")
        return enhanced_df

    def generate_temporal_features(self, df):
        """Generate temporal features with guaranteed creation of essential features"""
        print("\n[3/5] Generating robust temporal features...")
        start = time.time()
        
        # Store reference to dataframe
        self.df = df.copy()

        # 1. Ensure basic essential features exist
        self._create_essential_features(df)

        # 2. Create difference features safely (e.g., winner_* - loser_*)
        self._create_safe_difference_features(df)

        # 3. Sort by date to ensure correct temporal ordering
        df = df.sort_values('tourney_date').reset_index(drop=True)

        # 4. Initialize player history tracking
        player_history = defaultdict(lambda: {
            'matches': deque(maxlen=50),
            'last_match_date': None,
            'surface_stats': defaultdict(lambda: {'wins': 0, 'matches': 0})
        })

        # 5. Define base features we want to compute
        base_features = [
            'days_since_last', 'match_count_30d', 'match_count_90d',
            'win_rate_5', 'win_rate_10', 'win_rate_20',
            'momentum_5', 'momentum_10', 'momentum_20',
            'surface_win_rate', 'surface_match_count',
            'rest_quality', 'opponent_quality_5'
        ]

        # Initialize all possible feature columns to avoid KeyError later
        for prefix in ['winner_', 'loser_']:
            for feat in base_features:
                if feat not in df.columns:
                    df[prefix + feat] = np.nan

        # 6. Compute temporal features row-by-row
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building features"):
            match_date = row['tourney_date']
            surface = row['surface']

            for role in ['winner', 'loser']:
                player_id = row[f'{role}_id']
                ph = player_history[player_id]

                # a. Days since last match
                if ph['last_match_date']:
                    days_since = (match_date - ph['last_match_date']).days
                    df.at[idx, f'{role}_days_since_last'] = days_since
                else:
                    df.at[idx, f'{role}_days_since_last'] = 365  # Default value

                # b. Match counts
                recent_matches = [m for m in ph['matches'] 
                                if (match_date - m['date']).days <= 90]
                df.at[idx, f'{role}_match_count_30d'] = len(
                    [m for m in recent_matches 
                    if (match_date - m['date']).days <= 30]
                )
                df.at[idx, f'{role}_match_count_90d'] = len(recent_matches)

                # c. Win rates (with fallbacks)
                for window in [5, 10, 20]:
                    col = f'{role}_win_rate_{window}'
                    if len(ph['matches']) >= window:
                        window_matches = list(ph['matches'])[-window:]
                        df.at[idx, col] = sum(m['won'] for m in window_matches) / window
                    else:
                        df.at[idx, col] = 0.5  # Neutral default

                # d. Momentum (simplified to always exist)
                for window in [5, 10, 20]:
                    col = f'{role}_momentum_{window}'
                    if len(ph['matches']) >= window:
                        window_matches = list(ph['matches'])[-window:]
                        df.at[idx, col] = sum(
                            (i+1)/window * (1 if m['won'] else -1) 
                            for i, m in enumerate(window_matches)
                        )
                    else:
                        df.at[idx, col] = 0  # Neutral default

                # e. Surface stats
                surface_stats = ph['surface_stats'][surface]
                df.at[idx, f'{role}_surface_win_rate'] = (
                    surface_stats['wins'] / surface_stats['matches']
                    if surface_stats['matches'] > 0 else 0.5
                )
                df.at[idx, f'{role}_surface_match_count'] = surface_stats['matches']

                # f. Rest quality (simplified)
                if ph['matches']:
                    last_opponent_rank = ph['matches'][-1]['opponent_rank_points'] or 1000
                    df.at[idx, f'{role}_rest_quality'] = 1 / (1 + np.log1p(last_opponent_rank))
                else:
                    df.at[idx, f'{role}_rest_quality'] = 0.5

            # Update histories AFTER processing both players
            for role in ['winner', 'loser']:
                player_id = row[f'{role}_id']
                ph = player_history[player_id]
                ph['matches'].append({
                    'date': match_date,
                    'won': (role == 'winner'),
                    'opponent_rank_points': row[f'loser_rank_points'] if role == 'winner' else row[f'winner_rank_points'],
                    'surface': surface
                })
                if role == 'winner':
                    player_history[player_id]['surface_stats'][surface]['wins'] += 1
                player_history[player_id]['surface_stats'][surface]['matches'] += 1
                player_history[player_id]['last_match_date'] = match_date

        # 7. Create difference features ONLY for columns that exist
        diff_features = []
        for feat in base_features:
            winner_col = f'winner_{feat}'
            loser_col = f'loser_{feat}'
            if winner_col in df.columns and loser_col in df.columns:
                df[f'{feat}_diff'] = df[winner_col] - df[loser_col]
                diff_features.append(f'{feat}_diff')

        # 8. Special combined features (with existence checks)
        if all(c in df.columns for c in ['loser_days_since_last', 'winner_days_since_last']):
            df['rest_advantage'] = (df['loser_days_since_last'] - df['winner_days_since_last']) / 7
        else:
            df['rest_advantage'] = 0  # Fallback if not available

        if all(c in df.columns for c in ['loser_match_count_30d', 'winner_match_count_30d']):
            df['fatigue_diff'] = df['loser_match_count_30d'] - df['winner_match_count_30d']
        else:
            df['fatigue_diff'] = 0

        if all(c in df.columns for c in ['winner_surface_match_count', 'loser_surface_match_count']):
            df['surface_experience_diff'] = df['winner_surface_match_count'] - df['loser_surface_match_count']
        else:
            df['surface_experience_diff'] = 0

        # 9. Final feature selection
        all_potential_features = diff_features + [
            'rest_advantage', 'fatigue_diff', 'surface_experience_diff',
            'rank_diff', 'rank_ratio', 'age_diff', 'age_ratio', 'exp_diff'
        ]
        
        existing_features = [f for f in all_potential_features if f in df.columns]
        self.features = self._select_features_with_validation(df, existing_features)
        
        print(f"✅ Generated {len(self.features)} temporal features in {time.time()-start:.2f}s")
        print(f"Selected features: {self.features[:10]}...")
        print("🔍 DEBUG - Final features:", self.features)
        if not self.features:
            raise ValueError("No features were selected in generate_temporal_features.")
        return df

    def _create_essential_features(self, df):
        """Ensure fundamental features always exist"""
        # Rank features
        if 'winner_rank_points' in df.columns and 'loser_rank_points' in df.columns:
            df['rank_diff'] = df['winner_rank_points'] - df['loser_rank_points']
            df['rank_ratio'] = (df['winner_rank_points'] + 1) / (df['loser_rank_points'] + 1)
        
        # Age features
        if 'winner_age' in df.columns and 'loser_age' in df.columns:
            df['age_diff'] = df['winner_age'] - df['loser_age']
            df['age_ratio'] = (df['winner_age'] + 1) / (df['loser_age'] + 1)
        
        # Experience features
        if 'winner_clay_exp' in df.columns and 'loser_clay_exp' in df.columns:
            df['exp_diff'] = df['winner_clay_exp'] - df['loser_clay_exp']

    def _create_safe_difference_features(self, df):
        """Create difference features only when base features exist"""
        # List of possible prefix pairs
        prefixes = [
            ('winner_', 'loser_'),
            ('player1_', 'player2_')  # Add other naming conventions if needed
        ]
        
        # Features we might want to difference
        diff_candidates = [
            'days_since_last', 'match_count_30d', 'match_count_90d',
            'win_rate_5', 'win_rate_10', 'win_rate_20',
            'momentum_5', 'momentum_10', 'momentum_20',
            'surface_win_rate', 'surface_match_count'
        ]
        
        for feat in diff_candidates:
            for winner_prefix, loser_prefix in prefixes:
                winner_col = winner_prefix + feat
                loser_col = loser_prefix + feat
                
                if winner_col in df.columns and loser_col in df.columns:
                    df[f'{feat}_diff'] = df[winner_col] - df[loser_col]
                    break  # Use first valid prefix pair
    
    def _select_features_with_validation(self, df, candidate_features):
        """Select features that definitely exist and have variance"""
        print("🔍 DEBUG - _select_features_with_validation")
        print(f"Candidate features count: {len(candidate_features)}")
        print(f"Candidate features: {candidate_features}")
        # Filter to existing numeric features
        existing_features = [
            f for f in candidate_features 
            if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
        ]
        print(f"Existing numeric features count: {len(existing_features)}")
        print(f"Existing features: {existing_features}")
        # Remove near-constant features
        variability = df[existing_features].std()
        variable_features = variability[variability > 1e-6].index.tolist()
        
        if len(variable_features) < len(existing_features):
            constant_features = set(existing_features) - set(variable_features)
            print(f"⚠️ Dropping near-constant features: {constant_features}")
        
        # Ensure essential features are included
        essential_features = ['rank_diff', 'rank_ratio', 'exp_diff']
        for feat in essential_features:
            if feat in existing_features and feat not in variable_features:
                variable_features.append(feat)
        
        return variable_features

    def _select_existing_features(self, df, candidate_features):
        """Select only features that definitely exist"""
        # Basic feature selection from existing columns
        X = df[candidate_features].fillna(0)
        y = df['target']
        
        selector = SelectKBest(mutual_info_classif, k=min(20, len(candidate_features)))
        selector.fit(X, y)
        
        selected = [candidate_features[i] for i in selector.get_support(indices=True)]
        
        # Ensure key features are included
        for feat in ['rank_diff', 'rank_ratio']:
            if feat in candidate_features and feat not in selected:
                selected.append(feat)
        
        return selected


    def _select_features_with_temporal_safety(self, df, candidate_features):
        """Feature selection that respects temporal ordering"""
        from sklearn.feature_selection import mutual_info_classif
        
        # Temporal cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        feature_scores = defaultdict(float)
        
        for train_idx, val_idx in tscv.split(df):
            X_train = df.iloc[train_idx][candidate_features].fillna(0)
            y_train = df.iloc[train_idx]['target']
            
            # Get mutual information scores
            mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
            
            # Accumulate scores across folds
            for i, score in enumerate(mi_scores):
                feature_scores[candidate_features[i]] += score
        
        # Normalize scores
        max_score = max(feature_scores.values())
        normalized_scores = {f: s/max_score for f, s in feature_scores.items()}
        
        # Select top features (capped at 20)
        selected_features = sorted(normalized_scores.items(), key=lambda x: -x[1])
        selected_features = [f[0] for f in selected_features[:20]]
        
        # Ensure key features are included
        essential_features = ['rank_diff', 'rank_ratio', 'rest_advantage']
        for feat in essential_features:
            if feat not in selected_features and feat in candidate_features:
                selected_features.append(feat)
        
        return selected_features
        
        return df
    def _get_potential_features(self, df):
        """Identify all potential numeric features that could be used for modeling"""
        
        # Base features that can be directly used or used to create derived features
        base_features = [
            # Player attributes
            'winner_rank_points', 'loser_rank_points',
            'winner_age', 'loser_age',
            'winner_height', 'loser_height',
            'winner_recent_wins', 'loser_recent_wins',
            'winner_hard_exp', 'winner_clay_exp', 'winner_grass_exp',
            'loser_hard_exp', 'loser_clay_exp', 'loser_grass_exp',
            'h2h_wins',
            
            # Pre-computed features
            'rank_points_ratio'
        ]
        
        # Derived features that can be created from base features
        derived_features = [
            # Difference features
            'rank_diff', 'age_diff', 'height_diff', 'recent_wins_diff',
            'hard_exp_diff', 'clay_exp_diff', 'grass_exp_diff',
            
            # Ratio features
            'rank_ratio', 'age_ratio', 'height_ratio',
            'recent_wins_ratio', 'exp_ratio',
            
            # Interaction terms
            'rank_age_interaction', 'rank_exp_interaction',
            'age_exp_interaction'
        ]
        
        # Temporal features (these would be added by generate_temporal_features)
        temporal_features = [
            'days_since_last_diff', 'fatigue_diff', 'rest_advantage',
            'win_rate_diff_5', 'win_rate_diff_10', 'win_rate_diff_20',
            'momentum_diff_5', 'momentum_diff_10', 'momentum_diff_20',
            'winner_win_rate_5', 'winner_win_rate_10', 'winner_win_rate_20',
            'loser_win_rate_5', 'loser_win_rate_10', 'loser_win_rate_20',
            'winner_momentum_5', 'winner_momentum_10', 'winner_momentum_20',
            'loser_momentum_5', 'loser_momentum_10', 'loser_momentum_20'
        ]
        
        # Surface-specific features
        surface_features = [
            'surface_clay_advantage', 'surface_grass_advantage',
            'surface_hard_advantage', 'surface_exp_diff'
        ]
        
        # Tournament level features
        tourney_features = [
            'tourney_level_importance', 'round_importance'
        ]
        
        # Combine all potential features
        all_potential_features = (
            base_features + 
            derived_features + 
            temporal_features + 
            surface_features + 
            tourney_features
        )
        
        # Return only features that exist in the dataframe or can be derived
        existing_features = [f for f in all_potential_features if f in df.columns]
        
        # Add features that can be derived from existing columns
        if 'winner_rank_points' in df.columns and 'loser_rank_points' in df.columns:
            existing_features.extend(['rank_diff', 'rank_ratio'])
        if 'winner_age' in df.columns and 'loser_age' in df.columns:
            existing_features.extend(['age_diff', 'age_ratio'])
        if 'winner_height' in df.columns and 'loser_height' in df.columns:
            existing_features.extend(['height_diff'])
        if 'winner_recent_wins' in df.columns and 'loser_recent_wins' in df.columns:
            existing_features.extend(['recent_wins_diff'])
        if 'winner_clay_exp' in df.columns and 'loser_clay_exp' in df.columns:
            existing_features.extend(['clay_exp_diff'])
        
        # Remove duplicates
        existing_features = list(set(existing_features))
        
        # Filter to only numeric features
        numeric_features = [
            f for f in existing_features 
            if f in df.columns and 
            pd.api.types.is_numeric_dtype(df[f]) and 
            not df[f].isnull().all()
        ]
        
        return numeric_features

    def _select_features_with_stability(self, df, n_iter=5):
        """Feature selection with multiple runs for stability"""
        feature_scores = defaultdict(int)
        # Get potential features ensuring they are numeric and exist
        potential_features = self._get_potential_features(df)

        if not potential_features:
            raise ValueError("No potential features found for selection.")

        X = df[potential_features].fillna(0)
        y = df['target']

        for _ in range(n_iter):
            selector = SelectKBest(mutual_info_classif, k=min(20, len(potential_features)))
            selector.fit(X, y)
            for i in selector.get_support(indices=True):
                feature_scores[X.columns[i]] += 1

        # Sort and limit features
        selected_features = sorted(feature_scores.items(), key=lambda x: -x[1])[:20]
        selected_features = [f[0] for f in selected_features]

        if not selected_features:
            # Fallback to essential features if selection failed
            essential_features = ['rank_diff', 'form_diff', 'rest_advantage']
            selected_features = [f for f in essential_features if f in potential_features]
            if not selected_features:
                raise ValueError("No valid features could be selected including fallback essentials.")

        # Update preprocessor with selected features
        self.update_preprocessor(selected_features)
        print(f"✅ Selected {len(selected_features)} features: {selected_features}")
        return selected_features

    def _select_features_automatically(self, df):
        """Automatically select features using mutual information, excluding non-numeric columns"""
        print("\n🔍 Performing automated feature selection...")
        
        # Define expected numeric features
        numeric_features = [
            'rank_diff', 'rank_ratio', 'age_diff', 'exp_diff', 'form_diff',
            'win_rate_diff_5', 'win_rate_diff_10', 'win_rate_diff_20',
            'momentum_diff_5', 'momentum_diff_10', 'momentum_diff_20',
            'fatigue_diff', 'rest_advantage', 'rank_momentum_interaction',
            'fatigue_rest_interaction', 'winner_days_since_last', 'loser_days_since_last',
            'winner_fatigue', 'loser_fatigue',
            'winner_win_rate_5', 'winner_win_rate_10', 'winner_win_rate_20',
            'loser_win_rate_5', 'loser_win_rate_10', 'loser_win_rate_20',
            'winner_momentum_5', 'winner_momentum_10', 'winner_momentum_20',
            'loser_momentum_5', 'loser_momentum_10', 'loser_momentum_20'
        ]
        
        # Filter for columns that exist in the DataFrame and are numeric
        potential_features = [col for col in numeric_features if col in df.columns]
        non_numeric_cols = [col for col in df.columns if col in potential_features and df[col].dtype not in ['int64', 'float64']]
        
        if non_numeric_cols:
            print(f"⚠️ Warning: Non-numeric columns found in features: {non_numeric_cols}. Excluding them.")
            potential_features = [col for col in potential_features if col not in non_numeric_cols]
        
        if not potential_features:
            raise ValueError("No valid numeric features available for selection!")
        
        X = df[potential_features].fillna(0)
        y = df['target']
        
        # Select top 20 features
        selector = SelectKBest(mutual_info_classif, k=min(20, len(potential_features)))
        selector.fit(X, y)
        
        selected_features = [potential_features[i] for i in selector.get_support(indices=True)]
    
        # Update preprocessor with selected features
        self.update_preprocessor(selected_features)
        
        print(f"Selected {len(selected_features)} features: {selected_features}")
        return selected_features
    
    def _calculate_rolling_win_rate(self, history, window):
        """Vectorized calculation of rolling win rate"""
        if not history:
            return 0.5
        recent = history[-window:] if len(history) > window else history
        return np.mean([m['won'] for m in recent])

    def _calculate_momentum(self, history, window):
        """Vectorized calculation of momentum score"""
        if not history:
            return 0.0
        recent = history[-window:] if len(history) > window else history
        weights = np.linspace(0.1, 1.0, len(recent))
        outcomes = np.array([1 if m['won'] else -1 for m in recent])
        return np.sum(weights * outcomes)

    def create_robust_temporal_split(self, df, val_ratio=0.15, test_ratio=0.15):
        """Create temporally split data with buffer periods and stricter player isolation"""
        print("\n[4/5] Creating robust temporal split...")
        start = time.time()
        
        df = df.sort_values('tourney_date').reset_index(drop=True)
        total_size = len(df)
        
        # Calculate split points with buffer periods
        test_start_idx = int(total_size * (1 - test_ratio))
        val_start_idx = int(test_start_idx * (1 - val_ratio))
        
        # Add 6-month buffer between splits
        buffer_days = 180
        train_end_date = df.iloc[val_start_idx]['tourney_date'] - pd.Timedelta(days=buffer_days)
        val_end_date = df.iloc[test_start_idx]['tourney_date'] - pd.Timedelta(days=buffer_days)
        
        # Get indices for buffer periods
        train_initial = df[df['tourney_date'] <= train_end_date].copy()
        val_initial = df[(df['tourney_date'] > train_end_date) & 
                        (df['tourney_date'] <= val_end_date)].copy()
        test_initial = df[df['tourney_date'] > val_end_date].copy()
        
        print(f"Initial split sizes - Train: {len(train_initial)}, Val: {len(val_initial)}, Test: {len(test_initial)}")
        
        # Get players from val and test sets
        def get_players(df_split):
            return set(df_split['winner_id']).union(set(df_split['loser_id']))
        
        val_players = get_players(val_initial)
        test_players = get_players(test_initial)
        future_players = val_players.union(test_players)
        
        # Remove matches from training set that involve future players
        train_mask = (~train_initial['winner_id'].isin(future_players)) & \
                    (~train_initial['loser_id'].isin(future_players))
        
        train_final = train_initial[train_mask].copy()
        
        # Ensure minimum dataset sizes
        if len(train_final) < 10000 or len(val_initial) < 2000 or len(test_initial) < 2000:
            raise ValueError("Insufficient data after strict splitting. Consider reducing buffer sizes.")
        
        # Validate splits
        train_players = get_players(train_final)
        val_players_final = get_players(val_initial)
        test_players_final = get_players(test_initial)
        
        # Run leakage detection
        temporal_ok = self.leakage_detector.check_temporal_leakage(
            train_final['tourney_date'],
            val_initial['tourney_date'],
            test_initial['tourney_date']
        )
        
        player_ok = self.leakage_detector.check_player_leakage(
            train_players, val_players_final, test_players_final
        )
        
        if not temporal_ok or not player_ok:
            raise ValueError("Data leakage detected that couldn't be resolved automatically")
        
        print(f"✅ Robust temporal split created in {time.time()-start:.2f}s")
        print(f"Final sizes - Train: {len(train_final)}, Val: {len(val_initial)}, Test: {len(test_initial)}")
        print(f"Train date range: {train_final['tourney_date'].min()} to {train_final['tourney_date'].max()}")
        print(f"Val date range: {val_initial['tourney_date'].min()} to {val_initial['tourney_date'].max()}")
        print(f"Test date range: {test_initial['tourney_date'].min()} to {test_initial['tourney_date'].max()}")
        
        return train_final, val_initial, test_initial

    def create_adversarial_validation_set(self, train_df, val_df):
        """Create adversarial validation to detect distribution shift"""
        print("\n🔍 Running Adversarial Validation...")
        
        # Label train as 0, val as 1
        train_adv = train_df[self.features].copy()
        train_adv['is_val'] = 0
        val_adv = val_df[self.features].copy()
        val_adv['is_val'] = 1
        
        combined = pd.concat([train_adv, val_adv], ignore_index=True)
        
        # Train classifier to distinguish train from val
        X = combined[self.features].fillna(0)
        y = combined['is_val']
        
        from sklearn.model_selection import cross_val_score
        rf_adv = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(rf_adv, X, y, cv=3, scoring='roc_auc')
        
        print(f"Adversarial validation AUC: {np.mean(scores):.4f}")
        if np.mean(scores) > 0.6:
            print("⚠️ Significant distribution shift detected between train and validation!")
        else:
            print("✅ No significant distribution shift detected")
        
        return np.mean(scores)

    def train_ensemble_model(self, X_train, y_train, X_val, y_val):
        """Train ensemble model with robust feature handling"""
        print("\n[5/5] Training robust ensemble model...")
        start = time.time()
        
        # DEBUG: Show shapes and feature list
        print("🔍 DEBUG - train_ensemble_model:")
        print(f"Features list: {self.features}")
        print(f"X_train type: {type(X_train)}")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        
        # CRITICAL: Check for empty feature matrix FIRST
        if isinstance(X_train, np.ndarray) and X_train.shape[1] == 0:
            print("❌ CRITICAL ERROR: Empty feature matrix detected!")
            print("This indicates a preprocessing pipeline issue.")
            print("The preprocessing step is removing all features.")
            print("\nDEBUGGING INFO:")
            print(f"Expected features: {len(self.features)}")
            print(f"Actual features in X_train: {X_train.shape[1]}")
            print(f"Features list: {self.features}")
            
            # Try to get features directly from the dataframe
            print("\n🔍 Attempting to extract features directly from dataframe...")
            try:
                # Check if features exist in the original dataframe
                available_features = [f for f in self.features if f in self.df.columns]
                print(f"Available features in dataframe: {len(available_features)}")
                
                if available_features:
                    print("✅ Found features in dataframe, extracting directly...")
                    # Get the indices that were used for train/val split
                    # You'll need to modify this based on how you store your split indices
                    train_indices = getattr(self, 'train_indices', None)
                    val_indices = getattr(self, 'val_indices', None)
                    
                    if train_indices is not None and val_indices is not None:
                        X_train = self.df.iloc[train_indices][available_features].values
                        X_val = self.df.iloc[val_indices][available_features].values
                        self.features = available_features
                        print(f"✅ Extracted features directly: {X_train.shape}")
                    else:
                        raise ValueError("Cannot recover: train/val indices not available")
                else:
                    raise ValueError("No valid features found in dataframe")
                    
            except Exception as e:
                print(f"❌ Could not recover features: {str(e)}")
                raise ValueError("Preprocessing pipeline removed all features. Check your feature engineering and preprocessing steps.")
        
        # Convert back to DataFrame if needed and we have features
        if not isinstance(X_train, pd.DataFrame) and X_train.shape[1] > 0:
            print("⚠️ Converting NumPy array back to DataFrame")
            X_train = pd.DataFrame(X_train, columns=self.features[:X_train.shape[1]])
            X_val = pd.DataFrame(X_val, columns=self.features[:X_val.shape[1]])
        
        # Verify that features actually exist in the dataframe
        if X_train.shape[1] > 0:
            print("🔍 Validating feature existence in self.df...")
            missing_features = [f for f in self.features if f not in self.df.columns]
            if missing_features:
                print(f"❌ Critical Error: {len(missing_features)} features missing from dataframe!")
                print("Missing features:", missing_features)
                # Update features to only include existing ones
                existing_features = [f for f in self.features if f in self.df.columns]
                print(f"✅ Using only {len(existing_features)} valid features")
                
                if not existing_features:
                    raise ValueError("No valid features available for training")
                    
                # Update the features list and preprocessor
                self.features = existing_features
                self.update_preprocessor(self.features)
                
                # Re-process the data with valid features
                if isinstance(X_train, pd.DataFrame):
                    X_train = X_train[self.features]
                    X_val = X_val[self.features]
            else:
                print("✅ All selected features exist in dataframe")
        
        # Convert to numpy arrays if not already
        X_train = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
        X_val = np.array(X_val) if not isinstance(X_val, np.ndarray) else X_val

        # Final verification of feature dimensions
        if X_train.shape[1] == 0 or X_val.shape[1] == 0:
            print("❌ ERROR: Empty feature matrix detected during training!")
            print(f"Final features used: {self.features}")
            print(f"Available columns in df: {self.df.columns.tolist()}")
            print(f"X_train shape: {X_train.shape}")
            print(f"X_val shape: {X_val.shape}")
            raise ValueError("No features available for training - check preprocessing pipeline")
        
        print(f"✅ Feature validation passed: {X_train.shape[1]} features")
        
        # Feature weighting with bounds checking
        feature_weights = {
            'fatigue_diff': 0.8,
            'exp_diff': 0.9,
            'rank_ratio': 1.0,
            'rank_diff': 1.0,
            'form_diff': 1.1,
            'rest_advantage': 1.2
        }
        
        # Apply weights only to existing features
        for feat, weight in feature_weights.items():
            try:
                if feat in self.features:
                    idx = self.features.index(feat)
                    if idx < X_train.shape[1]:  # Check bounds
                        X_train[:, idx] = X_train[:, idx] * weight
                        X_val[:, idx] = X_val[:, idx] * weight
                        print(f"✅ Applied weight {weight} to feature {feat}")
            except (ValueError, IndexError):
                print(f"⚠️ Feature {feat} not available for weighting")

        # Base models with enhanced stability
        base_models = [
            ('xgb', XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=1.0,
                random_state=42,
                eval_metric='logloss',
                early_stopping_rounds=20,
                use_label_encoder=False
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                max_features='sqrt',
                min_samples_leaf=5,
                min_samples_split=10,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            )),
            ('logit', LogisticRegression(
                penalty='l2',
                C=0.5,
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ))
        ]
        
        # Meta model with calibration
        meta_model = CalibratedClassifierCV(
            LogisticRegression(penalty='l2', C=1.0, max_iter=1000),
            method='isotonic',
            cv=3
        )
        
        # Create stacking ensemble with input validation
        try:
            self.model = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_model,
                cv=3,
                passthrough=True,
                n_jobs=-1,
                stack_method='predict_proba'
            )
            
            print(f"🔍 Training ensemble with {X_train.shape[1]} features on {X_train.shape[0]} samples...")
            
            # Train with validation monitoring
            self.model.fit(X_train, y_train)
            
            # Evaluate
            self.evaluate_model_performance(X_train, y_train, X_val, y_val)
            print(f"✅ Ensemble trained successfully in {time.time()-start:.2f}s")
            return self.model
        except Exception as e:
            print(f"❌ Ensemble training failed: {str(e)}")
            print(f"X_train shape: {X_train.shape}")
            print(f"y_train shape: {y_train.shape}")
            print(f"Features: {self.features}")
            raise

    def evaluate_model_performance(self, X_train, y_train, X_val, y_val, output_dir='./model_evaluation', model_id=None):
        """Comprehensive model evaluation with prioritized metrics and visualizations.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            output_dir: Directory to save evaluation outputs
            model_id: Optional identifier for tracking model versions
        
        Returns:
            dict: Evaluation results with key metrics
        """
        print("\n=== Model Performance Evaluation ===")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results
        results = []
        
        # Evaluate on both training and validation sets
        for X, y, dataset_name in [(X_train, y_train, "Training"), (X_val, y_val, "Validation")]:
            # Predictions
            y_pred = self.model.predict(X)
            y_pred_proba = self.model.predict_proba(X)[:, 1]
            
            # Expected Calibration Error (ECE)
            prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=10)
            ece = np.mean(np.abs(prob_true - prob_pred))
            
            # Metrics
            metrics = {
                'Dataset': dataset_name,
                'ROC_AUC': roc_auc_score(y, y_pred_proba),
                'Log_Loss': log_loss(y, y_pred_proba),
                'ECE': ece,
                'Brier_Score': brier_score_loss(y, y_pred_proba),
                'Precision': precision_score(y, y_pred, zero_division=0),
                'Recall': recall_score(y, y_pred, zero_division=0),
                'F1_Score': f1_score(y, y_pred, zero_division=0),
                'Sample_Size': len(y)
            }
            results.append(metrics)
        
        # Display results
        results_df = pd.DataFrame(results)
        print("\n📊 Performance Metrics:")
        print(results_df.round(4).to_string(index=False))
        
        # Save metrics
        metrics_file = f"{output_dir}/performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate visualizations
        self._create_evaluation_visualizations(X_val, y_val, output_dir)
        
        # Track performance for model comparison
        if model_id:
            self._track_model_performance(results, model_id, output_dir)
        
        # Feature importance
        self._analyze_feature_importance(output_dir)
        
        return results

    def _create_evaluation_visualizations(self, X_val, y_val, output_dir):
        """Create evaluation visualizations (ROC, Calibration, Confusion Matrix).
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            output_dir: Directory to save plots
        """
        try:
            y_pred_proba = self.model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Create figure with three subplots
            plt.figure(figsize=(18, 5))
            
            # ROC Curve
            plt.subplot(1, 3, 1)
            fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_val, y_pred_proba):.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True)
            
            # Calibration Curve
            plt.subplot(1, 3, 2)
            prob_true, prob_pred = calibration_curve(y_val, y_pred_proba, n_bins=10)
            plt.plot(prob_pred, prob_true, 's-', label='Model Calibration')
            plt.plot([0, 1], [0, 1], 'k:', label='Perfect Calibration')
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title(f'Calibration Plot (ECE = {np.mean(np.abs(prob_true - prob_pred)):.3f})')
            plt.legend()
            plt.grid(True)
            
            # Confusion Matrix
            plt.subplot(1, 3, 3)
            cm = confusion_matrix(y_val, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Predicted Loser', 'Predicted Winner'],
                        yticklabels=['Actual Loser', 'Actual Winner'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/evaluation_plots.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")

    def _analyze_feature_importance(self, output_dir):
        """Analyze and visualize feature importance.
        
        Args:
            output_dir: Directory to save feature importance plot
        """
        print("\n📊 Feature Importance Analysis:")
        
        try:
            importances = []
            for name, estimator in self.model.named_estimators_.items():
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
            
            if importances:
                avg_importance = np.mean(importances, axis=0)
                feature_importance_df = pd.DataFrame({
                    'Feature': self.features,
                    'Importance': avg_importance
                }).sort_values('Importance', ascending=False)
                
                print(feature_importance_df.head(10).to_string(index=False))
                
                # Plot
                plt.figure(figsize=(10, 6))
                top_features = feature_importance_df.head(10)
                sns.barplot(x='Importance', y='Feature', data=top_features)
                plt.title('Top 10 Feature Importances')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"Could not analyze feature importance: {e}")

    def _track_model_performance(self, results, model_id, output_dir):
        """Track model performance across iterations for comparison.
        
        Args:
            results: Evaluation results from evaluate_model_performance
            model_id: Unique identifier for the model
            output_dir: Directory to save tracking data
        """
        tracking_file = f"{output_dir}/model_performance_history.json"
        
        # Load existing history or initialize
        if os.path.exists(tracking_file):
            with open(tracking_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        # Add current results
        val_metrics = next(r for r in results if r['Dataset'] == 'Validation')
        history.append({
            'model_id': model_id,
            'timestamp': datetime.now().isoformat(),
            'ROC_AUC': val_metrics['ROC_AUC'],
            'Log_Loss': val_metrics['Log_Loss'],
            'ECE': val_metrics['ECE'],
            'Brier_Score': val_metrics['Brier_Score']
        })
        
        # Save updated history
        with open(tracking_file, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        # Plot performance trends
        try:
            history_df = pd.DataFrame(history)
            plt.figure(figsize=(12, 6))
            
            metrics_to_plot = ['ROC_AUC', 'Log_Loss', 'ECE', 'Brier_Score']
            for metric in metrics_to_plot:
                plt.plot(history_df['timestamp'], history_df[metric], marker='o', label=metric)
            
            plt.xlabel('Timestamp')
            plt.ylabel('Metric Value')
            plt.title('Model Performance Trends')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/performance_trends.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create performance trend plot: {e}")
    

    def run_cross_validation(self, df, cv_folds=5):
        """Run time-series cross-validation"""
        print(f"\n🔄 Running {cv_folds}-fold Time Series Cross-Validation...")
        
        # Prepare features and target
        X = df[self.features].copy()
        y = df['target'].copy()
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        cv_scores = []
        fold_details = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"\n--- Fold {fold + 1}/{cv_folds} ---")
            
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Fit preprocessor on training fold only
            X_train_processed = self.preprocessor.fit_transform(X_train_fold)
            X_val_processed = self.preprocessor.transform(X_val_fold)
            
            # Train simple model for CV (faster than full ensemble)
            cv_model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            
            cv_model.fit(X_train_processed, y_train_fold)
            
            # Evaluate
            y_pred_proba = cv_model.predict_proba(X_val_processed)[:, 1]
            fold_auc = roc_auc_score(y_val_fold, y_pred_proba)
            
            cv_scores.append(fold_auc)
            fold_details.append({
                'fold': fold + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'auc': fold_auc
            })
            print(f"Fold {fold + 1}: AUC = {fold_auc:.4f}, Train size = {len(train_idx)}, Val size = {len(val_idx)}")
       
        # CV Summary
        cv_results_df = pd.DataFrame(fold_details)
        print(f"\n📊 Cross-Validation Summary:")
        print(cv_results_df.to_string(index=False))
        print(f"\nMean CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return cv_scores, cv_results_df

    def predict_match_outcome(self, winner_features, loser_features):
        """Predict outcome for a specific match"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Create feature vector (winner - loser differences)
        feature_dict = {}
        for feature in self.features:
            if feature.endswith('_diff') or feature.endswith('_advantage') or feature.endswith('_interaction'):
                # Already computed differences
                feature_dict[feature] = winner_features.get(feature, 0)
            else:
                # Compute difference
                winner_val = winner_features.get(feature.replace('_diff', ''), 0)
                loser_val = loser_features.get(feature.replace('_diff', ''), 0)
                feature_dict[feature] = winner_val - loser_val
        
        # Create DataFrame and preprocess
        match_df = pd.DataFrame([feature_dict])
        match_processed = self.preprocessor.transform(match_df)
        
        # Predict
        win_probability = self.model.predict_proba(match_processed)[0, 1]
        predicted_winner = 1 if win_probability > 0.5 else 0
        
        return {
            'winner_win_probability': win_probability,
            'loser_win_probability': 1 - win_probability,
            'predicted_winner': predicted_winner,
            'confidence': abs(win_probability - 0.5) * 2
        }

    def save_model(self, filepath):
        """Save trained model and preprocessor"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'features': self.features,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model and preprocessor"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.features = model_data['features']
        
        print(f"✅ Model loaded from {filepath}")
        print(f"Model trained on: {model_data.get('timestamp', 'Unknown')}")

    def generate_prediction_report(self, test_df, output_dir='./tennis_predictions'):
        """Generate comprehensive prediction report"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n📋 Generating Prediction Report...")
        
        # Prepare test data
        X_test = test_df[self.features].copy()
        y_test = test_df['target'].copy()
        
        # Preprocess
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Predictions
        y_pred = self.model.predict(X_test_processed)
        y_pred_proba = self.model.predict_proba(X_test_processed)[:, 1]
        
        # Comprehensive metrics
        report = {
            'model_performance': {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'log_loss': log_loss(y_test, y_pred_proba)
            },
            'test_set_info': {
                'total_matches': len(test_df),
                'date_range': f"{test_df['tourney_date'].min()} to {test_df['tourney_date'].max()}",
                'unique_players': len(set(test_df['winner_id']).union(set(test_df['loser_id']))),
                'class_distribution': y_test.value_counts().to_dict()
            }
        }
        
        # Confidence analysis
        confidence_scores = np.abs(y_pred_proba - 0.5) * 2
        report['confidence_analysis'] = {
            'mean_confidence': np.mean(confidence_scores),
            'high_confidence_matches': np.sum(confidence_scores > 0.7),
            'low_confidence_matches': np.sum(confidence_scores < 0.3)
        }
        
        # Performance by confidence level
        high_conf_mask = confidence_scores > 0.7
        if np.sum(high_conf_mask) > 0:
            high_conf_accuracy = accuracy_score(y_test[high_conf_mask], y_pred[high_conf_mask])
            report['confidence_analysis']['high_confidence_accuracy'] = high_conf_accuracy
        
        # Save detailed predictions
        predictions_df = test_df.copy()
        predictions_df['predicted_winner_wins'] = y_pred
        predictions_df['win_probability'] = y_pred_proba
        predictions_df['confidence'] = confidence_scores
        predictions_df['correct_prediction'] = (y_pred == y_test).astype(int)
        
        predictions_df.to_csv(f"{output_dir}/detailed_predictions.csv", index=False)
        
        # Save report
        import json
        with open(f"{output_dir}/prediction_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate visualizations
        self._create_evaluation_visualizations(X_test_processed, y_test, output_dir)
        
        print(f"✅ Prediction report saved to {output_dir}/")
        print(f"Test Accuracy: {report['model_performance']['accuracy']:.4f}")
        print(f"Test ROC-AUC: {report['model_performance']['roc_auc']:.4f}")
        
        return report

    def _create_prediction_visualizations(self, y_true, y_pred_proba, output_dir):
        """Create prediction visualization plots with thread-safe backend"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Set non-interactive backend
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_curve, confusion_matrix
            import seaborn as sns
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_true, y_pred_proba):.3f}')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True)
            
            # Prediction Distribution
            plt.subplot(1, 3, 2)
            plt.hist(y_pred_proba[y_true == 0], bins=20, alpha=0.7, label='Actual Loser Wins', density=True)
            plt.hist(y_pred_proba[y_true == 1], bins=20, alpha=0.7, label='Actual Winner Wins', density=True)
            plt.xlabel('Predicted Probability')
            plt.ylabel('Density')
            plt.title('Prediction Distribution')
            plt.legend()
            plt.grid(True)
            
            # Confidence Distribution
            plt.subplot(1, 3, 3)
            confidence = np.abs(y_pred_proba - 0.5) * 2
            plt.hist(confidence, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Prediction Confidence')
            plt.ylabel('Frequency')
            plt.title('Confidence Distribution')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/prediction_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Confusion Matrix
            y_pred = (y_pred_proba > 0.5).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Predicted Loser Win', 'Predicted Winner Win'],
                        yticklabels=['Actual Loser Win', 'Actual Winner Win'])
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")

    def main(self):
        """Main execution pipeline with comprehensive validation"""
        print("🎾 Enhanced Tennis Match Prediction System")
        print("=" * 50)
        # Initialize predictor
        predictor = EnhancedTennisPredictor()
        try:
            # Load and validate data
            df = predictor.load_data('clay_matches.csv')
            
            # Create match-level features
            df_features = predictor.create_match_features(df)
            
            # Generate temporal features
            df_final = predictor.generate_temporal_features(df_features)
            
            # DEBUG: Add this check after feature generation
            print("\n🔍 Post-generation Feature Check:")
            print(f"Total features available: {len(predictor.features)}")
            print(f"Features list: {predictor.features}")
            print(f"DataFrame shape: {df_final.shape}")
            print(f"Feature columns in df: {[c for c in predictor.features if c in df_final.columns]}")
            
            # *** ADD THIS: Setup preprocessor with the temporal features ***
            print("\n🔧 Setting up preprocessor with temporal features...")
            predictor.update_preprocessor(predictor.features, df_final)
            
            # Create robust temporal splits
            train_df, val_df, test_df = predictor.create_robust_temporal_split(df_final)
            
            # Adversarial validation check
            adv_score = predictor.create_adversarial_validation_set(train_df, val_df)
            
            # Prepare training data
            X_train = train_df[predictor.features].copy()
            y_train = train_df['target'].copy()
            X_val = val_df[predictor.features].copy()
            y_val = val_df['target'].copy()
            
            # DEBUG: Add this check before training
            print("\n🔍 Pre-training Feature Check:")
            print(f"X_train shape: {X_train.shape}")
            print(f"X_val shape: {X_val.shape}")
            print(f"Feature columns in X_train: {X_train.columns.tolist()}")
            print(f"Feature columns in X_val: {X_val.columns.tolist()}")
            
            # *** REPLACE THIS SECTION: Use the new preprocess_features method ***
            print("\n🔧 Preprocessing features...")
            X_train_processed = predictor.preprocess_features(X_train, fit=True)
            X_val_processed = predictor.preprocess_features(X_val, fit=False)
            
            # DEBUG: Check processed shapes
            print(f"X_train_processed shape: {X_train_processed.shape}")
            print(f"X_val_processed shape: {X_val_processed.shape}")
            
            # Train ensemble model
            model = predictor.train_ensemble_model(X_train_processed, y_train, X_val_processed, y_val)
            
            # Cross-validation
            cv_scores, cv_results = predictor.run_cross_validation(train_df)
            
            # Final test evaluation
            print("\n🏆 Final Test Set Evaluation")
            test_report = predictor.generate_prediction_report(test_df)
            
            # Save model
            model_path = f"tennis_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            predictor.save_model(model_path)
            print("\n✅ Pipeline completed successfully!")
            print(f"Final Test Accuracy: {test_report['model_performance']['accuracy']:.4f}")
            print(f"Final Test ROC-AUC: {test_report['model_performance']['roc_auc']:.4f}")
            print(f"Cross-Validation AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            
            # Example prediction
            print("\n🔮 Example Prediction:")
            sample_match = test_df.iloc[0]
            winner_features = {feat: sample_match[f'winner_{feat.split("_")[0]}'] if f'winner_{feat.split("_")[0]}' in sample_match else 0 for feat in predictor.features}
            loser_features = {feat: sample_match[f'loser_{feat.split("_")[0]}'] if f'loser_{feat.split("_")[0]}' in sample_match else 0 for feat in predictor.features}
            try:
                prediction = predictor.predict_match_outcome(winner_features, loser_features)
                print(f"Winner win probability: {prediction['winner_win_probability']:.3f}")
                print(f"Prediction confidence: {prediction['confidence']:.3f}")
            except Exception as e:
                print(f"Could not generate example prediction: {e}")
                
            return predictor, test_report
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

if __name__ == "__main__":
    predictor = EnhancedTennisPredictor()
    predictor, report = predictor.main()