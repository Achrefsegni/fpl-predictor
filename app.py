import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="FPL Predictor Pro - GW8",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# English translations
TRANSLATIONS = {
    'title': "⚽ FPL Predictor Pro",
    'subtitle': "🔮 AI-Powered Points Predictions for Gameweek 8",
    'navigation': "🎮 Navigation",
    'player_search': "🔍 Player Search",
    'player_placeholder': "ex: Salah, Haaland...",
    'filters': "🎯 Advanced Filters",
    'filter_team': "Filter by team:",
    'filter_position': "Filter by position:",
    'all_teams': "All",
    'all_positions': "All",
    'apply_filters': "🔍 Apply Filters",
    'view_filtered': "👥 View Filtered Players",
    'view_top10': "🏆 View Top 10",
    'searching': "🔍 Searching for",
    'player_not_found': "❌ Player not found. Try another name.",
    'player_card': "👤 Player",
    'team': "🏟️ Team",
    'cost': "💰 Cost",
    'position': "📊 Position",
    'stats': "📈 Current Statistics",
    'form': "Recent form",
    'total_points': "Total points",
    'ppg': "Points per game",
    'goals': "Goals",
    'assists': "Assists",
    'minutes': "Minutes",
    'match_context': "📅 Match Context",
    'location': "Location",
    'difficulty': "Difficulty",
    'opponent': "Opponent",
    'home': "🏠 Home",
    'away': "✈️ Away",
    'prediction': "🔮 Prediction for GW8",
    'interpretation': "💡 Interpretation",
    'excellent_choice': "Excellent choice! High point potential this week.",
    'good_choice': "Good choice! Solid performance expected.",
    'decent_choice': "Decent choice. Average performance expected.",
    'risky_choice': "Risky choice. Limited performance expected.",
    'filtered_players': "👥 Filtered Players",
    'no_players': "❌ No players match the selected criteria.",
    'players_found': "✅ {} players found",
    'top_predictions': "🏆 Top 10 Predictions",
    'how_to_use': "💡 How to use:",
    'usage_steps': [
        "🔍 Search for a specific player",
        "🎯 Filters to view all players from a team/position",
        "🏆 Top 10 for the best predictions",
        "👥 Filtered Players to explore by criteria"
    ],
    'goalkeeper': "Goalkeeper",
    'defender': "Defender",
    'midfielder': "Midfielder",
    'forward': "Forward",
    'confidence': "🎯 Prediction Confidence",
    'value': "💰 Value",
    'risk': "⚖️ Risk",
    'advanced_analysis': "📊 Advanced Analysis",
    'detailed_stats': "📊 View Detailed Stats",
    'add_favorites': "❤️ Add to Favorites",
    'advanced_stats': "📈 Advanced Statistics",
    'global_stats': "📊 Global Stats",
    'total_players': "Total players",
    'active_players': "Active players",
    'avg_points': "Average points",
    'discover_gems': "💎 Discover Hidden Gems",
    'hidden_gems': "💎 Hidden Gems",
    'team_analysis': "📊 My Team Analysis",
    'enter_team_id': "🔢 Enter your FPL Team ID:",
    'team_id_placeholder': "ex: 1234567",
    'analyze_team': "🔍 Analyze My Team",
    'total_predicted_points': "🎯 Total Predicted Points",
    'team_players': "👥 Your Team Players",
    'best_captain': "⭐ Best Captain",
    'team_risks': "⚠️ Team Risks",
    'no_team_data': "❌ No team data found",
    'team_analysis_title': "📊 Complete FPL Team Analysis",
    'suggested_transfers': "🔄 Suggested Transfers",
    'transfer_out': "➖ Transfer Out",
    'transfer_in': "➕ Transfer In",
    'expected_points_gain': "📈 Expected Points Gain",
    'make_transfer': "🔄 Make This Transfer",
    'no_transfers_suggested': "✅ No transfers suggested - your team is optimal!",
    'transfer_reason': "💡 Reason"
}

# Modern FPL-style CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .player-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .player-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }
    
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .prediction-badge {
        font-size: 1.8rem;
        font-weight: 800;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .top-player {
        background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
        color: #000;
    }
    
    .good-player {
        background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%);
        color: white;
    }
    
    .avg-player {
        background: linear-gradient(135deg, #2196F3 0%, #21CBF3 100%);
        color: white;
    }
    
    .poor-player {
        background: linear-gradient(135deg, #ff9800 0%, #ffb74d 100%);
        color: white;
    }
    
    .transfer-suggestion {
        background: linear-gradient(135deg, #a8e6cf 0%, #dcedc1 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        border-left: 5px solid #4caf50;
    }
    
    .transfer-warning {
        background: linear-gradient(135deg, #ffd3b6 0%, #ffaaa5 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        border-left: 5px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)

class FPLPredictor:
    def __init__(self):
        self.players_df = None
        self.teams_df = None
        self.fixtures_df = None
        self.model = None
        self.scaler = StandardScaler()
        self.all_player_names = []
        
    def load_all_fpl_data(self):
        """Load FPL data"""
        try:
            bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
            response = requests.get(bootstrap_url, timeout=10)
            data = response.json()
            
            self.players_df = pd.DataFrame(data['elements'])
            self.teams_df = pd.DataFrame(data['teams'])
            
            fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"
            fixtures_response = requests.get(fixtures_url, timeout=10)
            self.fixtures_df = pd.DataFrame(fixtures_response.json())
            
            self.all_player_names = self.players_df['web_name'].tolist()
            return True
        except Exception as e:
            st.error(f"Loading error: {e}")
            return False
    
    def clean_numeric_data(self, series):
        """Clean numeric data"""
        if pd.api.types.is_numeric_dtype(series):
            return series
        series_clean = series.replace([None, 'None', 'null', ''], '0')
        try:
            return pd.to_numeric(series_clean, errors='coerce')
        except:
            return pd.Series([0] * len(series), index=series.index)
    
    def create_features(self):
        """Create features for model training"""
        # Data cleaning
        critical_columns = ['form', 'points_per_game', 'goals_scored', 'assists', 'minutes', 
                           'total_points', 'influence', 'creativity', 'threat', 'ict_index', 'bps',
                           'clean_sheets', 'saves', 'goals_conceded', 'penalties_saved']
        
        for col in critical_columns:
            if col in self.players_df.columns:
                self.players_df[col] = self.clean_numeric_data(self.players_df[col])
        
        # Feature engineering
        self.players_df['cost'] = self.players_df['now_cost'] / 10
        self.players_df['points_per_minute'] = self.players_df['total_points'] / self.players_df['minutes']
        self.players_df['points_per_minute'] = self.players_df['points_per_minute'].replace([np.inf, -np.inf], 0).fillna(0)
        
        self.players_df['goal_involvement_per_90'] = (self.players_df['goals_scored'] + self.players_df['assists']) / (self.players_df['minutes'] / 90)
        self.players_df['goal_involvement_per_90'] = self.players_df['goal_involvement_per_90'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Defensive contribution
        self.players_df['defensive_contribution'] = 0.0
        
        # For defenders
        defender_mask = self.players_df['element_type'] == 2
        if defender_mask.any():
            self.players_df.loc[defender_mask, 'defensive_contribution'] += (
                self.players_df.loc[defender_mask, 'clean_sheets'] * 4 / 
                (self.players_df.loc[defender_mask, 'minutes'] / 90)
            )
            
        # For goalkeepers
        gk_mask = self.players_df['element_type'] == 1
        if gk_mask.any():
            self.players_df.loc[gk_mask, 'defensive_contribution'] += (
                self.players_df.loc[gk_mask, 'saves'] / 3 / 
                (self.players_df.loc[gk_mask, 'minutes'] / 90)
            )
            self.players_df.loc[gk_mask, 'defensive_contribution'] += (
                self.players_df.loc[gk_mask, 'clean_sheets'] * 4 / 
                (self.players_df.loc[gk_mask, 'minutes'] / 90)
            )
        
        self.players_df['defensive_contribution'] = self.players_df['defensive_contribution'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Minutes stability
        self.players_df['minutes_ratio'] = self.players_df['minutes'] / (7 * 90)
        self.players_df['minutes_ratio'] = self.players_df['minutes_ratio'].clip(0, 1)
        
        # Team features
        team_strength = dict(zip(self.teams_df['id'], self.teams_df['strength']))
        team_attack = dict(zip(self.teams_df['id'], self.teams_df['strength_attack_home']))
        team_defense = dict(zip(self.teams_df['id'], self.teams_df['strength_defence_home']))
        self.players_df['team_strength'] = self.players_df['team'].map(team_strength)
        self.players_df['team_attack'] = self.players_df['team'].map(team_attack)
        self.players_df['team_defense'] = self.players_df['team'].map(team_defense)
        
        # Match context for GW8 - IMPROVED FIXTURE LOADING
        next_gw_fixtures = self.fixtures_df[self.fixtures_df['event'] == 8]
        
        # If no GW8 fixtures found, try to find the next available gameweek
        if len(next_gw_fixtures) == 0:
            available_gws = sorted(self.fixtures_df['event'].unique())
            if len(available_gws) > 0:
                next_gw = available_gws[0]  # Use the next available GW
                next_gw_fixtures = self.fixtures_df[self.fixtures_df['event'] == next_gw]
                st.info(f"⚠️ Using Gameweek {next_gw} fixtures (GW8 not available yet)")
        
        team_difficulty = {}
        home_advantage = {}
        opponent_mapping = {}
        
        for _, fixture in next_gw_fixtures.iterrows():
            team_h = fixture['team_h']
            team_a = fixture['team_a']
            
            team_difficulty[team_h] = fixture['team_h_difficulty']
            team_difficulty[team_a] = fixture['team_a_difficulty']
            home_advantage[team_h] = 1
            home_advantage[team_a] = 0
            
            # Opponent mapping
            opponent_mapping[team_h] = team_a
            opponent_mapping[team_a] = team_h
        
        self.players_df['next_opponent_difficulty'] = self.players_df['team'].map(team_difficulty)
        self.players_df['is_home'] = self.players_df['team'].map(home_advantage)
        self.players_df['next_opponent'] = self.players_df['team'].map(opponent_mapping)
        self.players_df['difficulty_factor'] = (6 - self.players_df['next_opponent_difficulty']) / 5
        
        # Fill missing values for teams without fixtures
        self.players_df['next_opponent_difficulty'] = self.players_df['next_opponent_difficulty'].fillna(3)
        self.players_df['is_home'] = self.players_df['is_home'].fillna(0)
        self.players_df['next_opponent'] = self.players_df['next_opponent'].fillna(0)
    
    def prepare_features(self):
        """Prepare features for the model"""
        feature_columns = [
            'points_per_game', 'form', 'points_per_minute', 'goal_involvement_per_90', 
            'minutes_ratio', 'team_attack', 'team_defense', 'cost', 'difficulty_factor', 'is_home',
            'defensive_contribution'
        ]
        
        available_features = [col for col in feature_columns if col in self.players_df.columns]
        X = self.players_df[available_features].copy()
        
        for col in X.columns:
            X[col] = self.clean_numeric_data(X[col])
            X[col] = X[col].fillna(0)
            X[col] = X[col].replace([np.inf, -np.inf], 0)
        
        return X, available_features
    
    def train_model(self):
        """Train XGBoost model"""
        X, feature_names = self.prepare_features()
        y = self.players_df['points_per_game'].copy()
        
        self.model = xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        valid_players = self.players_df['minutes'] > 90
        X_filtered = X[valid_players]
        y_filtered = y[valid_players]
        
        if len(X_filtered) == 0:
            return False
        
        self.model.fit(X_filtered, y_filtered)
        return True
    
    def search_player(self, player_name):
        """Search for a player"""
        if not player_name or len(player_name.strip()) < 2:
            return pd.DataFrame()
        player_name_lower = player_name.lower()
        matches = self.players_df[
            self.players_df['web_name'].str.lower().str.contains(player_name_lower, na=False)
        ]
        return matches
    
    def search_players_autocomplete(self, query):
        """Search for players starting with the query"""
        if not query or len(query.strip()) < 1:
            return []
        
        query_lower = query.lower()
        matches = [name for name in self.all_player_names 
                  if name.lower().startswith(query_lower)]
        
        if not matches:
            matches = [name for name in self.all_player_names 
                      if query_lower in name.lower()]
        
        return matches[:10]
    
    def is_player_available(self, player):
        """Check if player is available to play"""
        try:
            is_available = True
            
            if 'chance_of_playing_next_round' in player:
                chance = player['chance_of_playing_next_round']
                
                if chance is None or pd.isna(chance):
                    pass
                elif isinstance(chance, (int, float)):
                    if chance == 0:
                        is_available = False
                elif isinstance(chance, str):
                    chance_lower = chance.lower().strip()
                    if chance_lower in ['0', '0%', 'none', 'null']:
                        is_available = False
            
            return is_available
            
        except Exception:
            return True
    
    def predict_player(self, player_name):
        """Predict player points - SIMPLIFIED: Model Prediction = Final Prediction"""
        matches = self.search_player(player_name)
        
        if len(matches) == 0:
            return None
        
        player = matches.iloc[0]
        
        if not self.is_player_available(player):
            return self.format_unavailable_player(player)
        
        X, feature_names = self.prepare_features()
        
        player_mask = self.players_df['id'] == player['id']
        if not player_mask.any():
            return None
        
        player_features = X[player_mask].iloc[0].values.reshape(1, -1)
        
        # SIMPLIFIED: Model Prediction = Final Prediction
        predicted_points = self.model.predict(player_features)[0]
        
        # Realistic limits
        predicted_points = np.clip(predicted_points, 1, 8)
        
        return self.format_prediction_result(player, predicted_points)
    
    def format_unavailable_player(self, player):
        """Format response for unavailable player"""
        opponent_id = player.get('next_opponent', 0)
        
        # IMPROVED OPPONENT DISPLAY
        if opponent_id and opponent_id != 0:
            opponent_name = self.get_team_name(opponent_id)
            is_home = player.get('is_home', 0)
            opponent_display = f"{opponent_name} (H)" if is_home == 1 else f"{opponent_name} (A)"
        else:
            opponent_display = "Fixture TBD"
        
        return {
            'player': player,
            'predicted_points': 0.0,
            'team_name': self.get_team_name(player['team']),
            'opponent_name': opponent_display,
            'is_home': player.get('is_home', 0),
            'unavailable': True
        }
    
    def format_prediction_result(self, player, predicted_points):
        """Format prediction result"""
        opponent_id = player.get('next_opponent', 0)
        
        # IMPROVED OPPONENT DISPLAY
        if opponent_id and opponent_id != 0:
            opponent_name = self.get_team_name(opponent_id)
            is_home = player.get('is_home', 0)
            opponent_display = f"{opponent_name} (H)" if is_home == 1 else f"{opponent_name} (A)"
        else:
            opponent_display = "Fixture TBD"

        return {
            'player': player,
            'predicted_points': predicted_points,
            'team_name': self.get_team_name(player['team']),
            'opponent_name': opponent_display,
            'is_home': player.get('is_home', 0),
            'unavailable': False
        }
    
    def get_team_name(self, team_id):
        """Return team name"""
        if self.teams_df is None:
            return "Unknown"
        team_mapping = dict(zip(self.teams_df['id'], self.teams_df['name']))
        return team_mapping.get(team_id, 'Unknown')
    
    def get_position_name(self, position_code):
        """Return position name"""
        positions = {1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Forward"}
        return positions.get(position_code, "Unknown")
    
    def get_filtered_players(self, team_filter, position_filter):
        """Filter players according to criteria"""
        filtered_players = self.players_df.copy()
        
        if team_filter != "All":
            team_id = self.teams_df[self.teams_df['name'] == team_filter]['id'].values
            if len(team_id) > 0:
                filtered_players = filtered_players[filtered_players['team'] == team_id[0]]
        
        if position_filter != "All":
            position_map = {"Goalkeeper": 1, "Defender": 2, "Midfielder": 3, "Forward": 4}
            position_code = position_map.get(position_filter)
            if position_code:
                filtered_players = filtered_players[filtered_players['element_type'] == position_code]
        
        return filtered_players
    
    def get_hidden_gems(self, max_ownership=5, min_minutes=180, top_n=5):
        """Find underestimated players"""
        if 'selected_by_percent' not in self.players_df.columns:
            return pd.DataFrame()
        
        ownership_clean = pd.to_numeric(self.players_df['selected_by_percent'], errors='coerce').fillna(0)
        
        gems = self.players_df[
            (ownership_clean < max_ownership) & 
            (self.players_df['minutes'] > min_minutes) &
            (self.players_df['form'] > 0)
        ].nlargest(top_n, 'form')
        
        return gems

    def get_user_team_data(self, team_id):
        """Get user team data"""
        try:
            if not team_id or not team_id.isdigit():
                st.error("❌ Invalid team ID")
                return None
                
            team_url = f"https://fantasy.premierleague.com/api/entry/{team_id}/"
            team_response = requests.get(team_url, timeout=10)
            
            if team_response.status_code != 200:
                st.error(f"❌ Team not found. Check your Team ID: {team_id}")
                return None
            
            team_data = team_response.json()
            
            current_gw = 7
            picks_url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{current_gw}/picks/"
            picks_response = requests.get(picks_url, timeout=10)
            
            if picks_response.status_code != 200:
                st.error("❌ Cannot retrieve team players")
                return None
            
            picks_data = picks_response.json()
            
            return {
                'team_info': team_data,
                'picks': picks_data
            }
        except Exception as e:
            st.error(f"❌ Loading error: {str(e)}")
            return None

    def analyze_user_team(self, team_id):
        """Analyze user team and return predictions"""
        team_data = self.get_user_team_data(team_id)
        
        if not team_data:
            return None
        
        analysis = {
            'team_name': team_data['team_info']['name'],
            'total_predicted_points': 0,
            'players': [],
            'best_captain': None,
            'risks': [],
            'player_count': 0,
            'formation': self.detect_formation(team_data['picks']['picks'])
        }
        
        for pick in team_data['picks']['picks']:
            player_id = pick['element']
            player_data = self.players_df[self.players_df['id'] == player_id]
            
            if len(player_data) > 0:
                player = player_data.iloc[0]
                player_name = player['web_name']
                
                prediction = self.predict_player(player_name)
                
                if prediction:
                    opponent_display = prediction['opponent_name']

                    player_analysis = {
                        'id': player_id,
                        'name': player_name,
                        'position': self.get_position_name(player['element_type']),
                        'position_code': player['element_type'],
                        'team': self.get_team_name(player['team']),
                        'predicted_points': prediction['predicted_points'],
                        'is_captain': pick.get('is_captain', False),
                        'is_vice_captain': pick.get('is_vice_captain', False),
                        'multiplier': 2 if pick.get('is_captain', False) else 1,
                        'opponent': opponent_display,
                        'difficulty': player.get('next_opponent_difficulty', 3),
                        'cost': player.get('now_cost', 0) / 10,
                        'unavailable': not self.is_player_available(player)
                    }
                    
                    analysis['players'].append(player_analysis)
                    analysis['total_predicted_points'] += player_analysis['predicted_points'] * player_analysis['multiplier']
                    analysis['player_count'] += 1
                    
                    # Check for risks
                    if not self.is_player_available(player):
                        chance = player.get('chance_of_playing_next_round', 'Unknown')
                        if chance == 0:
                            analysis['risks'].append(f"🏥 {player_name} - INJURED (0% chance)")
                        else:
                            analysis['risks'].append(f"⚠️ {player_name} - UNAVAILABLE")
                    
                    # Performance risks
                    if self.is_player_available(player):
                        if player.get('minutes', 0) < 90 and player.get('form', 0) < 2:
                            analysis['risks'].append(f"📉 {player_name} - Limited minutes & poor form")
                        
                        if player_analysis['predicted_points'] < 2.0:
                            analysis['risks'].append(f"🔻 {player_name} - Low predicted points ({player_analysis['predicted_points']:.1f})")
        
        if analysis['players']:
            available_players = [p for p in analysis['players'] if not p['unavailable']]
            if available_players:
                analysis['best_captain'] = max(available_players, key=lambda x: x['predicted_points'])
        
        return analysis

    def detect_formation(self, picks):
        """Detect team formation from player positions"""
        position_count = {1: 0, 2: 0, 3: 0, 4: 0}
        
        for pick in picks:
            player_id = pick['element']
            player_data = self.players_df[self.players_df['id'] == player_id]
            if len(player_data) > 0:
                position = player_data.iloc[0]['element_type']
                position_count[position] += 1
        
        formations = {
            (1, 3, 4, 3): "3-4-3",
            (1, 3, 5, 2): "3-5-2", 
            (1, 4, 3, 3): "4-3-3",
            (1, 4, 4, 2): "4-4-2",
            (1, 4, 5, 1): "4-5-1",
            (1, 5, 3, 2): "5-3-2",
            (1, 5, 4, 1): "5-4-1"
        }
        
        formation_key = (position_count[1], position_count[2], position_count[3], position_count[4])
        return formations.get(formation_key, "Custom")

    def suggest_transfers(self, team_analysis, max_suggestions=3):
        """Suggest transfers to improve team"""
        suggestions = []
        
        if not team_analysis or 'players' not in team_analysis:
            return suggestions
        
        current_players = team_analysis['players']
        
        # Find weak players (low predicted points or unavailable)
        weak_players = []
        for player in current_players:
            if player['unavailable'] or player['predicted_points'] < 2.5:
                weak_players.append(player)
        
        # If no obviously weak players, find the lowest performers
        if not weak_players:
            weak_players = sorted(current_players, key=lambda x: x['predicted_points'])[:2]
        
        for weak_player in weak_players[:max_suggestions]:
            # Find better alternatives in same position
            position = weak_player['position_code']
            current_cost = weak_player['cost']
            
            # Find better players in same position with similar cost
            alternatives = self.players_df[
                (self.players_df['element_type'] == position) &
                (self.players_df['now_cost'] / 10 <= current_cost + 1.0) &  # Allow slight budget increase
                (~self.players_df['id'].isin([p['id'] for p in current_players]))  # Not already in team
            ]
            
            # Get predictions for alternatives
            alternative_predictions = []
            for _, alt_player in alternatives.iterrows():
                alt_prediction = self.predict_player(alt_player['web_name'])
                if alt_prediction and alt_prediction['predicted_points'] > weak_player['predicted_points'] + 0.5:
                    alternative_predictions.append({
                        'player': alt_player,
                        'prediction': alt_prediction,
                        'improvement': alt_prediction['predicted_points'] - weak_player['predicted_points']
                    })
            
            # Sort by improvement and take best
            alternative_predictions.sort(key=lambda x: x['improvement'], reverse=True)
            
            if alternative_predictions:
                best_alternative = alternative_predictions[0]
                suggestions.append({
                    'transfer_out': weak_player,
                    'transfer_in': {
                        'name': best_alternative['player']['web_name'],
                        'team': self.get_team_name(best_alternative['player']['team']),
                        'position': self.get_position_name(best_alternative['player']['element_type']),
                        'predicted_points': best_alternative['prediction']['predicted_points'],
                        'cost': best_alternative['player']['now_cost'] / 10,
                        'improvement': best_alternative['improvement']
                    },
                    'reason': "Player unavailable" if weak_player['unavailable'] else "Low performance"
                })
        
        return suggestions

def get_translation(key):
    """Return translation for a given key"""
    return TRANSLATIONS.get(key, key)

def display_player_prediction(prediction, predictor):
    """Display player prediction"""
    if not prediction:
        return
    
    player = prediction['player']
    predicted_points = prediction['predicted_points']
    team_name = prediction['team_name']
    opponent_name = prediction['opponent_name']
    is_home = prediction['is_home']
    unavailable = prediction.get('unavailable', False)
    
    st.markdown("---")
    
    # Prediction badge
    if unavailable:
        badge_class = "poor-player"
        badge_text = "❌ 0.0 points (Unavailable)"
    elif predicted_points >= 6:
        badge_class = "top-player"
        badge_text = f"🚀 {predicted_points:.1f} pts 🚀"
    elif predicted_points >= 4:
        badge_class = "good-player"
        badge_text = f"🔥 {predicted_points:.1f} pts 🔥"
    elif predicted_points >= 2.5:
        badge_class = "avg-player"
        badge_text = f"⚡ {predicted_points:.1f} pts ⚡"
    else:
        badge_class = "poor-player"
        badge_text = f"📉 {predicted_points:.1f} pts 📉"
    
    st.markdown(f'<div class="prediction-badge {badge_class}">{badge_text}</div>', unsafe_allow_html=True)
    
    # Player card
    st.markdown('<div class="player-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### 👤 {player['web_name']}")
        st.markdown(f"**🏟️ Team:** {team_name}")
        st.markdown(f"**💰 Cost:** {player['cost']:.1f}M")
        st.markdown(f"**📊 Position:** {predictor.get_position_name(player['element_type'])}")
    
    with col2:
        # Quick stats
        st.metric("Form", f"{player['form']:.1f}")
        st.metric("PPG", f"{player['points_per_game']:.1f}")
        st.metric("Total Points", f"{player['total_points']}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Detailed stats
    st.markdown("---")
    st.markdown(f"#### {get_translation('stats')}")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Goals", f"{player['goals_scored']}")
        st.metric("Assists", f"{player['assists']}")
    
    with col4:
        st.metric("Minutes", f"{player['minutes']}")
        st.metric("Clean Sheets", f"{player.get('clean_sheets', 0)}")
    
    with col5:
        st.metric("ICT Index", f"{player.get('ict_index', 0):.1f}")
        st.metric("Influence", f"{player.get('influence', 0):.1f}")
    
    # Match context
    st.markdown("---")
    st.markdown(f"#### {get_translation('match_context')}")
    
    difficulty = player.get('next_opponent_difficulty', 3)
    difficulty_text = {1: "Very Easy 🟢", 2: "Easy 🟡", 3: "Medium 🟠", 4: "Difficult 🔴", 5: "Very Difficult 🛑"}
    location_text = get_translation('home') if is_home == 1 else get_translation('away')
    
    col6, col7, col8 = st.columns(3)
    with col6:
        st.info(f"**{get_translation('location')}:** {location_text}")
    with col7:
        st.info(f"**{get_translation('difficulty')}:** {difficulty_text.get(difficulty, 'Medium')}")
    with col8:
        st.info(f"**{get_translation('opponent')}:** vs {opponent_name}")
    
    # Interpretation
    st.markdown("---")
    st.markdown(f"#### {get_translation('interpretation')}")
    
    if unavailable:
        st.error("💡 **Player unavailable - consider transferring out**")
    elif predicted_points >= 6:
        st.success("💡 **Excellent choice! High point potential this week.**")
    elif predicted_points >= 4:
        st.success("💡 **Good choice! Solid performance expected.**")
    elif predicted_points >= 2.5:
        st.info("💡 **Decent choice. Average performance expected.**")
    else:
        st.warning("💡 **Risky choice. Limited performance expected.**")

def display_team_analysis_with_transfers(team_analysis, predictor):
    """Display team analysis with transfer recommendations"""
    if not team_analysis:
        return
    
    st.markdown(f"## {get_translation('team_analysis_title')}")
    
    # Team overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predicted Points", f"{team_analysis['total_predicted_points']:.1f}")
    with col2:
        st.metric("Formation", team_analysis['formation'])
    with col3:
        captain_name = team_analysis['best_captain']['name'] if team_analysis['best_captain'] else "None"
        st.metric("Best Captain", captain_name)
    with col4:
        risk_count = len(team_analysis['risks'])
        st.metric("Risks Identified", risk_count)
    
    # Team players
    st.markdown("---")
    st.markdown(f"### {get_translation('team_players')}")
    
    for player in team_analysis['players']:
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            with col1:
                badge = "© " if player['is_captain'] else "VC " if player['is_vice_captain'] else ""
                status = "❌ " if player['unavailable'] else ""
                st.write(f"**{badge}{status}{player['name']}** - {player['team']}")
            with col2:
                st.write(f"{player['position']}")
            with col3:
                st.write(f"{player['opponent']}")
            with col4:
                points_color = "🟢" if player['predicted_points'] >= 5 else "🟡" if player['predicted_points'] >= 3 else "🔴"
                st.write(f"**{points_color} {player['predicted_points']:.1f}**")
    
    # 🔄 TRANSFER RECOMMENDATIONS
    st.markdown("---")
    st.markdown(f"### {get_translation('suggested_transfers')}")
    
    transfer_suggestions = predictor.suggest_transfers(team_analysis)
    
    if transfer_suggestions:
        for i, suggestion in enumerate(transfer_suggestions):
            with st.container():
                if suggestion['transfer_out']['unavailable']:
                    st.markdown('<div class="transfer-warning">', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="transfer-suggestion">', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                
                with col1:
                    st.markdown(f"**{get_translation('transfer_out')}:**")
                    st.write(f"❌ {suggestion['transfer_out']['name']}")
                    st.write(f"{suggestion['transfer_out']['team']} | {suggestion['transfer_out']['position']}")
                    st.write(f"Predicted: {suggestion['transfer_out']['predicted_points']:.1f} pts")
                
                with col2:
                    st.markdown(f"**{get_translation('transfer_in')}:**")
                    st.write(f"✅ {suggestion['transfer_in']['name']}")
                    st.write(f"{suggestion['transfer_in']['team']} | {suggestion['transfer_in']['position']}")
                    st.write(f"Predicted: {suggestion['transfer_in']['predicted_points']:.1f} pts")
                
                with col3:
                    st.markdown(f"**{get_translation('expected_points_gain')}**")
                    st.markdown(f"**+{suggestion['transfer_in']['improvement']:.1f} pts**")
                
                with col4:
                    st.markdown(f"**{get_translation('transfer_reason')}**")
                    st.write(suggestion['reason'])
                    if st.button(f"🔄 Transfer", key=f"transfer_{i}"):
                        st.success(f"Transfer suggested: {suggestion['transfer_out']['name']} → {suggestion['transfer_in']['name']}")
                
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info(get_translation('no_transfers_suggested'))
    
    # Team risks
    if team_analysis['risks']:
        st.markdown("---")
        st.markdown(f"### {get_translation('team_risks')}")
        for risk in team_analysis['risks']:
            st.warning(risk)

def create_autocomplete_search():
    """Create search input with autocomplete functionality"""
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {get_translation('player_search')}")
    
    if 'search_suggestions' not in st.session_state:
        st.session_state.search_suggestions = []
    if 'show_suggestions' not in st.session_state:
        st.session_state.show_suggestions = False
    if 'selected_suggestion' not in st.session_state:
        st.session_state.selected_suggestion = None
    
    search_input = st.sidebar.text_input(
        get_translation('player_search') + ":",
        placeholder=get_translation('player_placeholder'),
        value=st.session_state.get('search_player', ''),
        key="player_search_input"
    )
    
    if search_input and len(search_input) >= 1:
        suggestions = st.session_state.predictor.search_players_autocomplete(search_input)
        st.session_state.search_suggestions = suggestions
        st.session_state.show_suggestions = len(suggestions) > 0
    else:
        st.session_state.search_suggestions = []
        st.session_state.show_suggestions = False
    
    if st.session_state.show_suggestions:
        st.sidebar.markdown("**Suggestions:**")
        for suggestion in st.session_state.search_suggestions:
            if st.sidebar.button(suggestion, key=f"sugg_{suggestion}"):
                st.session_state.search_player = suggestion
                st.session_state.show_suggestions = False
                st.session_state.selected_suggestion = suggestion
                st.rerun()
    
    if (search_input != st.session_state.get('search_player', '') or 
        st.session_state.selected_suggestion):
        if st.session_state.selected_suggestion:
            st.session_state.search_player = st.session_state.selected_suggestion
            st.session_state.selected_suggestion = None
        else:
            st.session_state.search_player = search_input

def main():
    if 'search_player' not in st.session_state:
        st.session_state.search_player = ""
    if 'team_id' not in st.session_state:
        st.session_state.team_id = ""
    if 'selected_suggestion' not in st.session_state:
        st.session_state.selected_suggestion = None
    
    st.markdown(f'<h1 class="main-header">{get_translation("title")}</h1>', unsafe_allow_html=True)
    st.markdown(f"### {get_translation('subtitle')}")
    
    st.sidebar.title(get_translation("navigation"))
    st.sidebar.markdown("---")
    
    if 'predictor' not in st.session_state:
        st.session_state.predictor = FPLPredictor()
        with st.spinner('🔄 Loading FPL data...'):
            if st.session_state.predictor.load_all_fpl_data():
                st.session_state.predictor.create_features()
                with st.spinner('🤖 Training AI model...'):
                    if st.session_state.predictor.train_model():
                        st.sidebar.success('✅ Model ready!')
                    else:
                        st.sidebar.error('❌ Training error')
            else:
                st.sidebar.error('❌ Data loading error')
    
    st.sidebar.markdown(f"## {get_translation('global_stats')}")
    if st.session_state.predictor.players_df is not None:
        total_players = len(st.session_state.predictor.players_df)
        active_players = len(st.session_state.predictor.players_df[st.session_state.predictor.players_df['minutes'] > 180])
        avg_points = st.session_state.predictor.players_df['total_points'].mean()
        
        st.sidebar.metric(get_translation('total_players'), total_players)
        st.sidebar.metric(get_translation('active_players'), active_players)
        st.sidebar.metric(get_translation('avg_points'), f"{avg_points:.0f}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {get_translation('team_analysis')}")
    
    st.sidebar.markdown("**Team ID Examples:**")
    st.sidebar.markdown("- `2673856` (Popular team)")
    st.sidebar.markdown("- `1234567` (Replace with yours)")
    
    team_id = st.sidebar.text_input(
        get_translation('enter_team_id'),
        placeholder=get_translation('team_id_placeholder'),
        value=st.session_state.team_id
    )
    
    if team_id != st.session_state.team_id:
        st.session_state.team_id = team_id
    
    analyze_team = st.sidebar.button(get_translation('analyze_team'))
    
    create_autocomplete_search()
    
    st.sidebar.markdown(f"## {get_translation('filters')}")
    
    team_options = [get_translation('all_teams')] + sorted(st.session_state.predictor.teams_df['name'].tolist())
    team_filter = st.sidebar.selectbox(get_translation('filter_team'), team_options)
    
    position_options = [get_translation('all_positions'), "Goalkeeper", "Defender", "Midfielder", "Forward"]
    position_filter = st.sidebar.selectbox(get_translation('filter_position'), position_options)
    
    apply_filters = st.sidebar.button(get_translation('apply_filters'))
    
    st.sidebar.markdown("---")
    if st.sidebar.button(get_translation('discover_gems')):
        gems = st.session_state.predictor.get_hidden_gems()
        
        st.markdown(f"## {get_translation('hidden_gems')}")
        
        if len(gems) > 0:
            for _, player in gems.iterrows():
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{player['web_name']}** - {st.session_state.predictor.get_team_name(player['team'])}")
                        ownership_display = player.get('selected_by_percent', 0)
                        if isinstance(ownership_display, str):
                            ownership_display = pd.to_numeric(ownership_display, errors='coerce')
                        st.write(f"Form: {player['form']:.1f} | Selected: {ownership_display:.1f}%")
                    with col2:
                        if st.button("🔍 View", key=f"gem_{player['id']}"):
                            st.session_state.search_player = player['web_name']
                            st.rerun()
        else:
            st.info("🔍 No hidden gems found")
    
    # Team Analysis Section
    if analyze_team and st.session_state.team_id:
        with st.spinner('🔍 Analyzing your FPL team...'):
            team_analysis = st.session_state.predictor.analyze_user_team(st.session_state.team_id)
            
            if team_analysis:
                display_team_analysis_with_transfers(team_analysis, st.session_state.predictor)
            else:
                st.error(get_translation('no_team_data'))
    
    # Player search section
    current_search = st.session_state.search_player
    if current_search and current_search.strip() and not analyze_team:
        with st.spinner(f'{get_translation("searching")} {current_search}...'):
            prediction = st.session_state.predictor.predict_player(current_search)
            
            if prediction:
                display_player_prediction(prediction, st.session_state.predictor)
            else:
                st.error(get_translation('player_not_found'))
    
    # Filtered players section
    if apply_filters and not analyze_team:
        st.markdown(f"## {get_translation('filtered_players')}")
        
        filtered_players = st.session_state.predictor.get_filtered_players(team_filter, position_filter)
        
        if len(filtered_players) == 0:
            st.warning(get_translation('no_players'))
        else:
            st.success(get_translation('players_found').format(len(filtered_players)))
            
            display_data = []
            for _, player in filtered_players.head(20).iterrows():
                display_data.append({
                    'Player': player['web_name'],
                    'Team': st.session_state.predictor.get_team_name(player['team']),
                    'Position': st.session_state.predictor.get_position_name(player['element_type']),
                    'Form': f"{player['form']:.1f}",
                    'PPG': f"{player['points_per_game']:.1f}",
                    'Total Points': player['total_points']
                })
            
            display_df = pd.DataFrame(display_data)
            st.dataframe(display_df, use_container_width=True)
    
    # Top 10 predictions
    st.sidebar.markdown("---")
    if st.sidebar.button(get_translation('view_top10')) and not analyze_team:
        st.markdown(f"## {get_translation('top_predictions')}")
        
        if st.session_state.predictor.model is None:
            st.error("❌ Model not available")
            return
        
        all_predictions = []
        
        for _, player in st.session_state.predictor.players_df.iterrows():
            if player['minutes'] > 90:
                try:
                    prediction = st.session_state.predictor.predict_player(player['web_name'])
                    if prediction and prediction['predicted_points'] > 0:
                        all_predictions.append({
                            'Player': player['web_name'],
                            'Team': st.session_state.predictor.get_team_name(player['team']),
                            'Position': st.session_state.predictor.get_position_name(player['element_type']),
                            'Predicted Points': prediction['predicted_points'],
                            'Form': player['form'],
                            'Cost': player['cost']
                        })
                except:
                    continue
        
        if all_predictions:
            top_df = pd.DataFrame(all_predictions)
            top_df = top_df.nlargest(10, 'Predicted Points')
            
            try:
                styled_df = top_df.style.format({
                    'Predicted Points': '{:.1f}',
                    'Form': '{:.1f}',
                    'Cost': '{:.1f}'
                }).background_gradient(subset=['Predicted Points'], cmap='YlOrRd')
                st.dataframe(styled_df, use_container_width=True)
            except:
                st.dataframe(top_df, use_container_width=True)
        else:
            st.warning("❌ No predictions available")
    
    st.markdown("---")
    st.markdown(f"### {get_translation('how_to_use')}")
    
    for step in get_translation('usage_steps'):
        st.markdown(f"- {step}")

if __name__ == "__main__":
    main()