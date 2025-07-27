import time
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from tennis_ML import TennisDataProcessor

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class FrenchOpenSimulator:
    def __init__(self, model_data=None):
        """
        Initialize the simulator with model data
        Args:
            model_data: Dictionary containing model, scaler, and features
        """
        if model_data:
            self.model = model_data['ensemble_model']
            self.scaler = model_data['scaler'] 
            self.features = model_data['features']
            print(f"✅ Model loaded with {len(self.features)} features: {list(self.features)}")
        else:
            raise ValueError("Model data must be provided")
        
        # French Open 2024 details
        self.tournament_info = {
            'tourney_id': 'roland_garros_2024',
            'surface': 'Clay',
            'tourney_level': 'G',  # Grand Slam
            'draw_size': 128,
            'best_of': 5,
            'start_date': '2024-05-26'  # French Open 2024 start date
        }
        
        # Round mappings
        self.round_names = {
            1: 'R128', 2: 'R64', 3: 'R32', 4: 'R16',
            5: 'QF', 6: 'SF', 7: 'F'
        }
    
    def load_player_data(self, filepath):
        """Load player data with historical stats up to French Open 2024"""
        print("📊 Loading player historical data...")
        
        # This should be your historical data up to May 2024
        df = pd.read_csv(filepath)
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
        
        # Filter to matches before French Open 2024
        cutoff_date = pd.to_datetime('2024-05-25')
        df_filtered = df[df['tourney_date'] <= cutoff_date].copy()
        
        print(f"   📈 Using {len(df_filtered)} matches up to {cutoff_date.date()}")
        return df_filtered
    
    def get_player_stats_at_date(self, player_id, historical_data, target_date):
        """Get player statistics at a specific date (French Open start)"""
        player_matches = historical_data[
            (historical_data['winner_id'] == player_id) | 
            (historical_data['loser_id'] == player_id)
        ].copy()
        
        # Filter to matches before target date
        player_matches = player_matches[player_matches['tourney_date'] < target_date]
        
        if len(player_matches) == 0:
            return self.get_default_player_stats()
        
        # Calculate basic stats
        wins = len(player_matches[player_matches['winner_id'] == player_id])
        total = len(player_matches)
        
        # Clay court specific stats
        clay_matches = player_matches[player_matches['surface'] == 'Clay']
        clay_wins = len(clay_matches[clay_matches['winner_id'] == player_id])
        clay_total = len(clay_matches)
        
        # Recent form (last 10 matches)
        recent_matches = player_matches.sort_values('tourney_date', ascending=False).head(10)
        recent_wins = len(recent_matches[recent_matches['winner_id'] == player_id])
        
        # Get latest ranking info
        latest_match = player_matches.iloc[-1]
        if latest_match['winner_id'] == player_id:
            rank = latest_match.get('winner_rank', 100)
            rank_points = latest_match.get('winner_rank_points', 0)
        else:
            rank = latest_match.get('loser_rank', 100)
            rank_points = latest_match.get('loser_rank_points', 0)
        
        # Calculate momentum (weighted recent performance)
        momentum_weights = [0.4, 0.3, 0.2, 0.1] if len(recent_matches) >= 4 else [1.0/len(recent_matches)] * len(recent_matches)
        recent_results = [(1 if match['winner_id'] == player_id else 0) for _, match in recent_matches.head(4).iterrows()]
        momentum = sum(w * r for w, r in zip(momentum_weights, recent_results)) if recent_results else 0.5
        
        # Calculate rank trend (simplified)
        rank_trend = 0.0
        if len(player_matches) >= 5:
            # Get ranking from matches in last 6 months
            six_months_ago = target_date - timedelta(days=180)
            recent_rank_matches = player_matches[player_matches['tourney_date'] >= six_months_ago]
            
            if len(recent_rank_matches) >= 2:
                # Simple trend: compare first and last rank in period
                first_match = recent_rank_matches.iloc[0]
                last_match = recent_rank_matches.iloc[-1]
                
                first_rank = first_match.get('winner_rank' if first_match['winner_id'] == player_id else 'loser_rank', rank)
                last_rank = last_match.get('winner_rank' if last_match['winner_id'] == player_id else 'loser_rank', rank)
                
                if pd.notna(first_rank) and pd.notna(last_rank):
                    rank_trend = (first_rank - last_rank) / 100  # Positive = improving
        
        # Calculate fatigue (matches in last 2 weeks)
        two_weeks_ago = target_date - timedelta(days=14)
        recent_activity = len(player_matches[player_matches['tourney_date'] >= two_weeks_ago])
        fatigue = min(recent_activity / 3, 1.0)  # Normalize to 0-1
        
        # Peak rank (best rank in dataset)
        all_ranks = []
        for _, match in player_matches.iterrows():
            if match['winner_id'] == player_id:
                match_rank = match.get('winner_rank')
            else:
                match_rank = match.get('loser_rank')
            if pd.notna(match_rank) and match_rank > 0:
                all_ranks.append(match_rank)
        
        peak_rank = min(all_ranks) if all_ranks else rank
        
        return {
            'matches_played': total,
            'win_rate': wins / total if total > 0 else 0.5,
            'surface_win_rate': clay_wins / clay_total if clay_total > 0 else 0.5,
            'recent_form': recent_wins / len(recent_matches) if len(recent_matches) > 0 else 0.5,
            'rank': rank if pd.notna(rank) else 100,
            'rank_points': rank_points if pd.notna(rank_points) else 0,
            'days_since_last': (target_date - player_matches['tourney_date'].max()).days,
            'momentum': momentum,
            'rank_trend': rank_trend,
            'fatigue': fatigue,
            'peak_rank': peak_rank
        }
    
    def get_default_player_stats(self):
        """Default stats for unknown players"""
        return {
            'matches_played': 0,
            'win_rate': 0.5,
            'surface_win_rate': 0.5,
            'recent_form': 0.5,
            'rank': 100,
            'rank_points': 0,
            'days_since_last': 30,
            'momentum': 0.5,
            'rank_trend': 0.0,
            'fatigue': 0.0,
            'peak_rank': 100
        }
    
    def create_match_features(self, player1_id, player2_id, round_num, historical_data, match_date):
        """Create features for a match between two players - ONLY the features the model was trained on"""
        
        # Get player stats at the time of French Open
        p1_stats = self.get_player_stats_at_date(player1_id, historical_data, match_date)
        p2_stats = self.get_player_stats_at_date(player2_id, historical_data, match_date)
        
        # Calculate H2H if available
        h2h_matches = historical_data[
            ((historical_data['winner_id'] == player1_id) & (historical_data['loser_id'] == player2_id)) |
            ((historical_data['winner_id'] == player2_id) & (historical_data['loser_id'] == player1_id))
        ]
        
        h2h_total = len(h2h_matches)
        h2h_p1_wins = len(h2h_matches[h2h_matches['winner_id'] == player1_id])
        h2h_win_rate = h2h_p1_wins / h2h_total if h2h_total > 0 else 0.5
        
        # Get additional stats for advanced features
        p1_momentum = p1_stats.get('momentum', p1_stats['recent_form'])
        p2_momentum = p2_stats.get('momentum', p2_stats['recent_form'])
        
        # Calculate ALL possible feature differences (we'll select the needed ones later)
        rank_diff = p1_stats['rank'] - p2_stats['rank']
        rank_points_diff = p1_stats['rank_points'] - p2_stats['rank_points']
        age_diff = 0  # Default since age data not available
        experience_diff = p1_stats['matches_played'] - p2_stats['matches_played']
        win_rate_diff = p1_stats['win_rate'] - p2_stats['win_rate']
        surface_win_rate_diff = p1_stats['surface_win_rate'] - p2_stats['surface_win_rate']
        form_diff = p1_stats['recent_form'] - p2_stats['recent_form']
        momentum_diff = p1_momentum - p2_momentum
        rest_diff = p2_stats['days_since_last'] - p1_stats['days_since_last']
        peak_rank_diff = p1_stats.get('peak_rank', p1_stats['rank']) - p2_stats.get('peak_rank', p2_stats['rank'])
        rank_trend_diff = p1_stats.get('rank_trend', 0) - p2_stats.get('rank_trend', 0)
        fatigue_diff = p1_stats.get('fatigue', 0) - p2_stats.get('fatigue', 0)
        
        # Surface specialization (how much better/worse on clay vs overall)
        p1_surface_spec = p1_stats['surface_win_rate'] - p1_stats['win_rate']
        p2_surface_spec = p2_stats['surface_win_rate'] - p2_stats['win_rate']
        surface_specialization = p1_surface_spec - p2_surface_spec
        surface_expertise_diff = p1_surface_spec - p2_surface_spec
        
        # Create ALL possible features that the training pipeline might use
        all_features = {
            # Basic differences
            'rank_diff': rank_diff,
            'rank_points_diff': rank_points_diff,
            'age_diff': age_diff,
            'experience_diff': experience_diff,
            'win_rate_diff': win_rate_diff,
            'surface_win_rate_diff': surface_win_rate_diff,
            'form_diff': form_diff,
            'momentum_diff': momentum_diff,
            'rest_diff': rest_diff,
            
            # Interaction features
            'rank_experience_interaction': rank_diff * experience_diff / 1000,
            'form_momentum_combo': form_diff * momentum_diff,
            'surface_specialization': surface_specialization,
            
            # Categorical features (encoded to match training)
            'surface_encoded': self.get_surface_encoding('Clay'),
            'tourney_level_encoded': self.get_tourney_level_encoding('G'),
            'round_encoded': self.get_round_encoding(round_num),
            
            # Tournament features
            'best_of': self.tournament_info['best_of'],
            'draw_size': self.tournament_info['draw_size'],
            
            # H2H features
            'h2h_matches': h2h_total,
            'h2h_win_rate': h2h_win_rate,
            
            # Advanced features
            'surface_expertise_diff': surface_expertise_diff,
            'fatigue_diff': fatigue_diff,
            'peak_rank_diff': peak_rank_diff,
            'rank_trend_diff': rank_trend_diff
        }
        
        return all_features
    
    def get_surface_encoding(self, surface):
        """Get surface encoding to match training data"""
        surface_map = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}
        return surface_map.get(surface, 1)  # Default to Clay
    
    def get_tourney_level_encoding(self, level):
        """Get tournament level encoding to match training data"""
        level_map = {'G': 0, 'M': 1, 'A': 2, 'C': 3, 'S': 4, 'F': 5}
        return level_map.get(level, 0)  # Default to Grand Slam
    
    def get_round_encoding(self, round_num):
        """Get round encoding to match training data"""
        round_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}  # R128=0, R64=1, etc.
        return round_map.get(round_num, 2)  # Default to R32
    
    def predict_match(self, player1_id, player2_id, round_num, historical_data, match_date):
        """Predict the outcome of a match"""
        
        # Create all possible features
        all_features = self.create_match_features(player1_id, player2_id, round_num, historical_data, match_date)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([all_features])
        
        # Get only the features that the model expects
        model_expects = list(self.features)
        print(f"🔍 Model expects {len(model_expects)} features: {model_expects}")
        print(f"📊 Available features: {list(feature_df.columns)}")
        
        # Create feature matrix with only the features the model was trained on
        try:
            # Only select features that the model expects AND we have available
            available_expected_features = [f for f in model_expects if f in feature_df.columns]
            missing_features = [f for f in model_expects if f not in feature_df.columns]
            
            if missing_features:
                print(f"⚠️  Missing features: {missing_features}")
                # Add missing features with default values
                for feature in missing_features:
                    feature_df[feature] = 0
            
            # Select features in the exact order the model expects
            X = feature_df[model_expects].copy()
            
            print(f"✅ Final feature shape: {X.shape}")
            print(f"✅ Feature order: {list(X.columns)}")
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict
            win_probability = self.model.predict_proba(X_scaled)[0, 1]
            winner = player1_id if win_probability > 0.5 else player2_id
            confidence = max(win_probability, 1 - win_probability)
            
            return {
                'winner': winner,
                'loser': player2_id if winner == player1_id else player1_id,
                'probability': win_probability,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"⚠️  Prediction error: {e}")
            print(f"Expected features: {model_expects}")
            print(f"Provided features: {list(feature_df.columns)}")
            
            # Fallback: random prediction
            return {
                'winner': player1_id,
                'loser': player2_id,
                'probability': 0.5,
                'confidence': 0.5
            }
    
    def simulate_tournament(self, draw_file, historical_data_file):
        """Simulate the entire French Open tournament"""
        print("🎾 Simulating French Open 2024...")
        
        # Load the draw
        draw = self.load_draw(draw_file)
        print(f"   📋 Starting with {len(draw)} players")
        
        # Load historical data
        historical_data = self.load_player_data(historical_data_file)
        
        # Simulate each round
        results = {}
        current_draw = draw.copy()
        
        for round_num in range(1, 8):  # 7 rounds in a Grand Slam
            round_name = self.round_names[round_num]
            
            # Check if we have enough players for this round
            if len(current_draw) < 2:
                print(f"   ✅ Tournament complete - not enough players for {round_name}")
                break
                
            print(f"\n🏆 Simulating {round_name}... ({len(current_draw)} players)")
            
            round_results = []
            next_round_players = []
            
            # Calculate match date (French Open progresses over 2 weeks)
            match_date = pd.to_datetime(self.tournament_info['start_date']) + timedelta(days=(round_num-1)*2)
            
            # Pair up players for matches
            matches_in_round = len(current_draw) // 2
            print(f"   🎯 Playing {matches_in_round} matches in {round_name}")
            
            for i in range(0, len(current_draw), 2):
                if i + 1 < len(current_draw):
                    player1 = current_draw[i]
                    player2 = current_draw[i + 1]
                    
                    # Predict match
                    prediction = self.predict_match(
                        player1['id'], player2['id'], round_num, 
                        historical_data, match_date
                    )
                    
                    # Find winner info
                    winner_info = player1 if prediction['winner'] == player1['id'] else player2
                    loser_info = player2 if prediction['winner'] == player1['id'] else player1
                    
                    match_result = {
                        'round': round_name,
                        'player1': player1,
                        'player2': player2,
                        'winner': winner_info,
                        'loser': loser_info,
                        'probability': prediction['probability'],
                        'confidence': prediction['confidence']
                    }
                    
                    round_results.append(match_result)
                    next_round_players.append(winner_info)
                    
                    print(f"   {player1['name']} vs {player2['name']} → {winner_info['name']} ({prediction['confidence']:.1%})")
                else:
                    # Odd number of players - bye to final player
                    print(f"   {current_draw[i]['name']} gets a bye")
                    next_round_players.append(current_draw[i])
            
            results[round_name] = round_results
            current_draw = next_round_players
            
            print(f"   ➡️  {len(next_round_players)} players advance to next round")
            
            # Tournament ends when we have a winner (1 player left)
            if len(current_draw) == 1:
                print(f"   🏆 Tournament Winner: {current_draw[0]['name']}")
                break
        
        # Print tournament summary
        self.print_tournament_summary(results)
        
        return results
    
    def load_draw(self, draw_file):
        """Load tournament draw from file"""
        try:
            with open(draw_file, 'r') as f:
                draw = json.load(f)
            return draw
        except FileNotFoundError:
            print("⚠️  Draw file not found. Creating sample draw...")
            return self.create_sample_draw()
    
    def create_sample_draw(self):
        """Create a sample draw for demonstration"""
        sample_players = [
            {'id': 'novak_djokovic', 'name': 'Novak Djokovic', 'seed': 1},
            {'id': 'carlos_alcaraz', 'name': 'Carlos Alcaraz', 'seed': 3},
            {'id': 'rafael_nadal', 'name': 'Rafael Nadal', 'seed': 'WC'},
            {'id': 'alexander_zverev', 'name': 'Alexander Zverev', 'seed': 4},
            {'id': 'stefanos_tsitsipas', 'name': 'Stefanos Tsitsipas', 'seed': 5},
            {'id': 'casper_ruud', 'name': 'Casper Ruud', 'seed': 7},
            {'id': 'andrey_rublev', 'name': 'Andrey Rublev', 'seed': 6},
            {'id': 'grigor_dimitrov', 'name': 'Grigor Dimitrov', 'seed': 10}
        ]
        
        print("   📝 Using sample 8-player draw for demonstration")
        return sample_players
    
    def print_tournament_summary(self, results):
        """Print a summary of the tournament results"""
        print(f"\n🏆 FRENCH OPEN 2024 SIMULATION SUMMARY")
        print("=" * 50)
        
        if 'F' in results and results['F']:
            champion = results['F'][0]['winner']
            runner_up = results['F'][0]['loser']
            final_confidence = results['F'][0]['confidence']
            
            print(f"🥇 CHAMPION: {champion['name']}")
            print(f"🥈 Runner-up: {runner_up['name']}")
            print(f"   Final confidence: {final_confidence:.1%}")
        
        print(f"\n📊 Round-by-round results:")
        for round_name, round_matches in results.items():
            print(f"\n{round_name}:")
            for match in round_matches:
                p1_name = match['player1']['name']
                p2_name = match['player2']['name']
                winner_name = match['winner']['name']
                conf = match['confidence']
                print(f"   {p1_name} vs {p2_name} → {winner_name} ({conf:.1%})")


def train_and_save_model():
    """Train a new model and save it"""
    
    print("🚀 Training model for French Open simulation...")
    
    # Initialize and train model
    processor = TennisDataProcessor()
    
    # Fix the bug in your original code
    results, processed_data = processor.process_full_pipeline('tennis_atp_2000_plus_matches.csv')
    
    # From your training output, these are the exact 12 features selected:
    selected_features = [
        'rank_diff', 'rank_points_diff', 'form_diff', 'surface_win_rate_diff',
        'rank_experience_interaction', 'peak_rank_diff', 'win_rate_diff',
        'momentum_diff', 'rest_diff', 'age_diff', 'draw_size', 'best_of'
    ]
    
    print(f"🎯 Using hardcoded selected features: {selected_features}")
    print(f"📊 Number of features: {len(selected_features)}")
    
    # Save model for future use with ONLY the selected features
    model_data = {
        'ensemble_model': results['ensemble_model'],
        'scaler': results['scaler'],
        'features': selected_features  # Use the exact 12 features that were selected
    }
    
    with open('french_open_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("✅ Model trained and saved to french_open_model.pkl")
    return model_data


def load_saved_model(force_retrain=False):
    """Load a previously saved model"""
    if force_retrain:
        print("🔄 Force retraining model...")
        return train_and_save_model()
        
    try:
        with open('french_open_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Validate that the model has the expected structure
        required_keys = ['ensemble_model', 'scaler', 'features']
        if not all(key in model_data for key in required_keys):
            print("⚠️  Saved model is corrupted. Retraining...")
            return train_and_save_model()
            
        print("✅ Model loaded from french_open_model.pkl")
        print(f"   Model features: {list(model_data['features'])}")
        return model_data
    except (FileNotFoundError, pickle.UnpicklingError, KeyError) as e:
        print(f"❌ Error loading saved model: {e}")
        print("Training new model...")
        return train_and_save_model()


def main():
    """Main function to run the simulation"""
    start_time = time.time()
    # Check if we want to force retrain the model
    force_retrain = input("🤔 Force retrain model? (y/N): ").lower().startswith('y')
    
    # Load or train model
    print("🔄 Loading model...")
    model_data = load_saved_model(force_retrain=force_retrain)
    
    if model_data is None:
        print("❌ Failed to load or train model")
        return
    
    # Initialize simulator
    simulator = FrenchOpenSimulator(model_data=model_data)
    
    # Run simulation
    tournament_results = simulator.simulate_tournament(
        draw_file='french_open_2024_draw.json',
        historical_data_file='tennis_atp_2000_plus_matches.csv'
    )
    
    # Save results
    with open('french_open_2024_simulation.json', 'w') as f:
        # Convert datetime objects to strings for JSON serialization
        def serialize_datetime(obj):
            if isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        json.dump(tournament_results, f, indent=2, default=serialize_datetime)
    print(f" Total time taken: {time.time() - start_time} seconds")
    print("\n✅ Simulation complete! Results saved to french_open_2024_simulation.json")


if __name__ == "__main__":
    main()