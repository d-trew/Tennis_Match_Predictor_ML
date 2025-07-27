import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import joblib
from tqdm.auto import tqdm
import time
import warnings
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import StackingClassifier
from  xgboost import XGBClassifier
# === Model Evaluation ===
# Best Parameters: {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 15, 'max_features': 0.8, 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 2, 'min_samples_split': 10, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 300, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}
# +----+------------+------------+-----------+-------------+----------+----------+
# |    | Dataset    |   Accuracy |   ROC AUC |   Precision |   Recall |       F1 |
# +====+============+============+===========+=============+==========+==========+
# |  0 | Training   |   0.86918  |  0.9534   |    0.872541 | 0.86918  | 0.868884 |
# +----+------------+------------+-----------+-------------+----------+----------+
# |  1 | Validation |   0.808452 |  0.897651 |    0.810054 | 0.808452 | 0.808204 |
# +----+------------+------------+-----------+-------------+----------+----------+

# 🔍 Feature Importances:
# +----+-----------------------+--------------+
# |    | Feature               |   Importance |
# +====+=======================+==============+
# |  8 | days_since_last_match |   0.438995   |
# +----+-----------------------+--------------+
# |  2 | recent_wins_diff      |   0.156055   |
# +----+-----------------------+--------------+
# |  0 | rank_diff             |   0.147731   |
# +----+-----------------------+--------------+
# |  6 | rank_ratio            |   0.115334   |
# +----+-----------------------+--------------+
# |  1 | exp_diff              |   0.0457218  |
# +----+-----------------------+--------------+
# |  7 | experience_gap        |   0.0378259  |
# +----+-----------------------+--------------+
# |  3 | age_diff              |   0.0272579  |
# +----+-----------------------+--------------+
# |  5 | momentum              |   0.0265022  |
# +----+-----------------------+--------------+
# |  4 | is_clay               |   0.00457668 |
# +----+-----------------------+--------------+

# ✓ Training completed in 117.37s

# Test ROC AUC: 0.9066
#               precision    recall  f1-score   support

#            0       0.79      0.86      0.82     10079
#            1       0.85      0.77      0.81     10079

#     accuracy                           0.82     20158
#    macro avg       0.82      0.82      0.82     20158
# Configure warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class EnhancedTennisPredictor:
    def __init__(self):
        self.features = None
        self.model = None
        self.preprocessor = make_pipeline(
            SimpleImputer(strategy='median'),
        )
        self.best_score = 0
        self.no_improvement_count = 0

    def load_data(self, filepath):
        """Optimized data loading with chunking for large files"""
        print("\n[1/4] Loading data...")
        start = time.time()
        
        try:
            # Use chunking for memory efficiency
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
        
        # Use numpy for faster operations
        winner_data = df.copy()
        loser_data = df.copy()
        
        # Vectorized column swapping
        swap_cols = ['id', 'rank_points', 'clay_exp', 'recent_wins', 'age']
        for col in swap_cols:
            if f'winner_{col}' in df.columns:
                winner_col = f'winner_{col}'
                loser_col = f'loser_{col}'
                loser_data[[winner_col, loser_col]] = loser_data[[loser_col, winner_col]].values
        
        # Concatenate with targets
        winner_data['target'] = 1
        loser_data['target'] = 0
        balanced_df = pd.concat([winner_data, loser_data], ignore_index=True)
        
        print(f"✓ Created {len(balanced_df)} records in {time.time()-start:.2f}s")
        return balanced_df

    def prepare_features(self, df):
        """Enhanced feature engineering"""
        print("\n[3/4] Preparing features...")
        start = time.time()
        
        # Core features with vectorized operations
        df['rank_diff'] = df['winner_rank_points'] - df['loser_rank_points']
        df['exp_diff'] = df['winner_clay_exp'] - df['loser_clay_exp']
        df['recent_wins_diff'] = df['winner_recent_wins'] - df['loser_recent_wins']
        df['age_diff'] = df['winner_age'] - df['loser_age']
        df['is_clay'] = df['surface'].eq('Clay').astype(int)
        
        # Advanced interaction features
        df['momentum'] = df['recent_wins_diff'] * df['exp_diff']
        df['rank_ratio'] = np.where(
            df['loser_rank_points'] == 0, 
            1, 
            df['winner_rank_points'] / df['loser_rank_points']
        )
        df['experience_gap'] = np.log1p(df['winner_clay_exp']) - np.log1p(df['loser_clay_exp'])
        
        df['relative_momentum'] = df['recent_wins_diff'] / (df['age_diff'].abs() + 1)  # Normalized by age gap
        df['rank_pressure'] = np.where(
            df['winner_rank_points'] > df['loser_rank_points'],
            df['winner_rank_points'] - df['loser_rank_points'],
            0  # Only consider when favorite is higher ranked
        )
        df['recovery_indicator'] = df['days_since_last_match'] * df['recent_wins_diff']  # Interaction term

        # Time-based features
        df['days_since_last_match'] = df.groupby('winner_id')['tourney_date'].diff().dt.days.fillna(0)
        
        self.features = [
            'rank_diff', 'exp_diff', 'recent_wins_diff', 
            'age_diff', 'is_clay', 'momentum', 'rank_ratio',
            'experience_gap', 'days_since_last_match'
        ]
        
        print(f"✓ Prepared {len(self.features)} features in {time.time()-start:.2f}s")
        return df

    def _early_stopping(self, current_score, patience=3, min_delta=0.001):
        """Early stopping implementation"""
        if current_score > (self.best_score + min_delta):
            self.best_score = current_score
            self.no_improvement_count = 0
            return False
        else:
            self.no_improvement_count += 1
            return self.no_improvement_count >= patience

    def train_model(self, X_train, y_train, X_val, y_val):
        """Enhanced training with early stopping and progress tracking"""
        print("\n[4/4] Training model...")
        start = time.time()
        
        # Optimized parameter space
        # param_dist = {
        #     'n_estimators': [200, 300],
        #     'max_depth': [10, 15],
        #     'min_samples_split': [5, 10],
        #     'max_features': ['sqrt', 0.8],
        #     'min_samples_leaf': [2],
        #     'bootstrap': [True, False]
        # }
        param_dist = {
            'n_estimators': [ 300],
            'max_depth': [15],
            'min_samples_split': [10],
            'max_features': [ 0.8],
            'min_samples_leaf': [2],
            'bootstrap': [True]
        }
            
        # Create progress bar
        pbar = tqdm(total=100, desc="Training Progress")
        
        # Custom callback for progress updates
        def update_progress():
            current_progress = min(
                (self.completed_fits / self.total_fits) * 100, 
                100
            )
            pbar.n = current_progress
            pbar.refresh()
        
        # Early stopping tracker
        self.best_score = 0
        self.no_improvement = 0
        self.completed_fits = 0
        self.total_fits = len(param_dist['n_estimators']) * \
                        len(param_dist['max_depth']) * \
                        len(param_dist['min_samples_split']) * \
                        len(param_dist['max_features']) * \
                        3  # cv folds
        
        # Custom scoring function
        def scorer(estimator, X, y):
            val_pred = estimator.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, val_pred)
            
            # Early stopping check
            if score > self.best_score + 0.001:
                self.best_score = score
                self.no_improvement = 0
            else:
                self.no_improvement += 1
                
            self.completed_fits += 1
            update_progress()
            
            if self.no_improvement >= 3:  # patience of 3
                raise ConvergenceWarning("Early stopping triggered")
                
            return score
        
        try:
            search = RandomizedSearchCV(
                estimator=RandomForestClassifier(
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                ),
                param_distributions=param_dist,
                n_iter=15,
                cv=3,
                scoring=scorer,
                verbose=0,
                random_state=42
            )
            
            search.fit(X_train, y_train)
            pbar.n = 100
            pbar.refresh()
            
        except ConvergenceWarning as e:
            print(f"\nEarly stopping: {str(e)}")
        finally:
            pbar.close()
        
        self.model = search.best_estimator_
        train_time = time.time() - start
        
        self._evaluate_model(X_train, y_train, X_val, y_val)
        print(f"\n✓ Training completed in {train_time:.2f}s")
        return self.model

    def _evaluate_model(self, X_train, y_train, X_test, y_test):
        """Enhanced model evaluation"""
        print("\n=== Model Evaluation ===")
        print(f"Best Parameters: {self.model.get_params()}")
        
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
        
        # Print formatted results
        print(pd.DataFrame(results).to_markdown(tablefmt="grid"))
        
        # Enhanced feature importance
        print("\n🔍 Feature Importances:")
        imp_df = pd.DataFrame({
            'Feature': self.features,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print(imp_df.to_markdown(tablefmt="grid"))

def main():
    try:
        predictor = EnhancedTennisPredictor()
        
        # 1. Load data
        df = predictor.load_data('processed_tennis_data.csv')
        
        # 2. Balance dataset
        balanced_df = predictor.create_balanced_dataset(df)
        
        # 3. Feature engineering
        processed_df = predictor.prepare_features(balanced_df)
        
        # 4. Temporal split with validation set
        train_val = processed_df[processed_df['tourney_date'] < '2021-06-01']
        test = processed_df[processed_df['tourney_date'] >= '2021-06-01']
        
        # Split train into train and validation
        train = train_val[train_val['tourney_date'] < '2021-01-01']
        val = train_val[train_val['tourney_date'] >= '2021-01-01']
        
        # Preprocess data
        X_train = predictor.preprocessor.fit_transform(train[predictor.features])
        y_train = train['target']
        X_val = predictor.preprocessor.transform(val[predictor.features])
        y_val = val['target']
        X_test = predictor.preprocessor.transform(test[predictor.features])
        y_test = test['target']
        
        # 5. Train and evaluate
        model = predictor.train_model(X_train, y_train, X_val, y_val)
        
        # Final test evaluation
        print("\n=== Final Test Evaluation ===")
        test_preds = model.predict(X_test)
        test_probs = model.predict_proba(X_test)[:, 1]
        print(f"Test ROC AUC: {roc_auc_score(y_test, test_probs):.4f}")
        print(classification_report(y_test, test_preds))
        
        # Save model
        joblib.dump(model, 'enhanced_tennis_model.pkl')
        print("\n✔ Model saved to enhanced_tennis_model.pkl")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")

if __name__ == "__main__":
    main()