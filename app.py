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
    page_title="FPL Predictor Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# English translations only
TRANSLATIONS = {
    'title': "⚽ FPL Predictor Pro",
    'subtitle': "🔮 Points Predictions for Gameweek 7",
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
    'prediction': "🔮 Prediction for GW7",
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

# Modern FPL-style CSS inspired by the reference image
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #37003c;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    /* Modern FPL Lineup Styles */
    .fpl-modern-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        min-height: 700px;
        position: relative;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .pitch-background {
        background: url('https://www.footballkitarchive.com/images/stadium-background.jpg');
        background-size: cover;
        background-position: center;
        border-radius: 15px;
        position: absolute;
        top: 20px;
        left: 20px;
        right: 20px;
        bottom: 20px;
        opacity: 0.1;
        z-index: 1;
    }
    
    .lineup-content {
        position: relative;
        z-index: 2;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    .team-header {
        text-align: center;
        color: white;
        margin-bottom: 30px;
    }
    
    .team-name {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 5px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .formation {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Player rows positioning */
    .players-row {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 25px;
        margin: 25px 0;
        flex-wrap: wrap;
    }
    
    .forwards-row {
        margin-top: 50px;
    }
    
    .midfielders-row {
        margin-top: 120px;
    }
    
    .defenders-row {
        margin-top: 120px;
    }
    
    .goalkeeper-row {
        margin-top: 120px;
    }
    
    /* Modern player cards */
    .modern-player-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 15px;
        min-width: 140px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        border: 3px solid transparent;
        position: relative;
        backdrop-filter: blur(10px);
    }
    
    .modern-player-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.25);
    }
    
    .modern-player-card.captain {
        border-color: #ffd700;
        background: linear-gradient(135deg, #fff9c4 0%, #ffffff 100%);
    }
    
    .modern-player-card.vice-captain {
        border-color: #c0c0c0;
    }
    
    .modern-player-card.unavailable {
        background: linear-gradient(135deg, #ffcdd2 0%, #ffffff 100%);
        opacity: 0.8;
    }
    
    .player-name {
        font-weight: 700;
        font-size: 14px;
        color: #37003c;
        margin-bottom: 5px;
        line-height: 1.2;
    }
    
    .player-team {
        font-size: 11px;
        color: #666;
        margin-bottom: 3px;
        font-weight: 600;
    }
    
    .player-fixture {
        font-size: 10px;
        color: #37003c;
        background: #e8f5e8;
        padding: 2px 6px;
        border-radius: 8px;
        margin-bottom: 5px;
        font-weight: 600;
    }
    
    .player-fixture.away {
        background: #fff3e0;
    }
    
    .player-points {
        font-size: 16px;
        font-weight: 700;
        color: #37003c;
        margin-top: 5px;
    }
    
    .player-points.high {
        color: #00c853;
    }
    
    .player-points.medium {
        color: #ff9800;
    }
    
    .player-points.low {
        color: #f44336;
    }
    
    .captain-badge {
        position: absolute;
        top: -8px;
        right: -8px;
        background: #ffd700;
        color: #37003c;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 10px;
        font-weight: 700;
    }
    
    .vice-captain-badge {
        position: absolute;
        top: -8px;
        right: -8px;
        background: #c0c0c0;
        color: #37003c;
        border-radius: 50%;
        width: 18px;
        height: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 9px;
        font-weight: 700;
    }
    
    .unavailable-badge {
        position: absolute;
        top: -8px;
        left: -8px;
        background: #f44336;
        color: white;
        border-radius: 50%;
        width: 18px;
        height: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 9px;
        font-weight: 700;
    }
    
    /* Bench section */
    .bench-section {
        margin-top: 40px;
        padding: 20px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    
    .bench-title {
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 15px;
        text-align: center;
    }
    
    .bench-players {
        display: flex;
        justify-content: center;
        gap: 15px;
        flex-wrap: wrap;
    }
    
    .bench-player-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 10px;
        min-width: 120px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    /* Stats section */
    .stats-container {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
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
    
    /* Responsive design */
    @media (max-width: 1200px) {
        .modern-player-card {
            min-width: 120px;
            padding: 12px;
        }
        
        .player-name {
            font-size: 13px;
        }
    }
    
    @media (max-width: 768px) {
        .fpl-modern-container {
            padding: 20px;
            min-height: 600px;
        }
        
        .players-row {
            gap: 15px;
            margin: 20px 0;
        }
        
        .modern-player-card {
            min-width: 100px;
            padding: 10px;
        }
        
        .player-name {
            font-size: 12px;
        }
        
        .team-name {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

class FPLWebPredictor:
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
            
            # Store all player names for autocomplete
            self.all_player_names = self.players_df['web_name'].tolist()
            
            return True
        except Exception as e:
            st.error(f"Loading error: {e}")
            return False
    
    def search_players_autocomplete(self, query):
        """Search for players starting with the query (case insensitive)"""
        if not query or len(query.strip()) < 1:
            return []
        
        query_lower = query.lower()
        matches = [name for name in self.all_player_names 
                  if name.lower().startswith(query_lower)]
        
        # If no starts-with matches, try contains
        if not matches:
            matches = [name for name in self.all_player_names 
                      if query_lower in name.lower()]
        
        return matches[:10]  # Return top 10 matches
    
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
        """Create features for the model"""
        # Data cleaning
        critical_columns = ['form', 'points_per_game', 'goals_scored', 'assists', 'minutes', 
                           'total_points', 'influence', 'creativity', 'threat', 'ict_index', 'bps']
        
        for col in critical_columns:
            if col in self.players_df.columns:
                self.players_df[col] = self.clean_numeric_data(self.players_df[col])
        
        # Basic features
        self.players_df['cost'] = self.players_df['now_cost'] / 10
        self.players_df['points_per_minute'] = self.players_df['total_points'] / self.players_df['minutes']
        self.players_df['points_per_minute'] = self.players_df['points_per_minute'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Offensive impact
        self.players_df['goal_involvement_per_90'] = (self.players_df['goals_scored'] + self.players_df['assists']) / (self.players_df['minutes'] / 90)
        self.players_df['goal_involvement_per_90'] = self.players_df['goal_involvement_per_90'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Consistency
        self.players_df['consistency_score'] = self.players_df['points_per_game'] / self.players_df['points_per_game'].max()
        
        # Minutes stability
        self.players_df['minutes_ratio'] = self.players_df['minutes'] / (6 * 90)  # Using GW6 data
        self.players_df['minutes_ratio'] = self.players_df['minutes_ratio'].clip(0, 1)
        
        # Team features
        team_strength = dict(zip(self.teams_df['id'], self.teams_df['strength']))
        team_attack = dict(zip(self.teams_df['id'], self.teams_df['strength_attack_home']))
        self.players_df['team_strength'] = self.players_df['team'].map(team_strength)
        self.players_df['team_attack'] = self.players_df['team'].map(team_attack)
        
        # Match context for GW7
        next_gw_fixtures = self.fixtures_df[self.fixtures_df['event'] == 7]
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
    
    def prepare_features(self):
        """Prepare features for the model"""
        feature_columns = [
            'points_per_game', 'form', 'points_per_minute', 'consistency_score',
            'goal_involvement_per_90', 'minutes_ratio', 'team_attack', 'cost',
            'difficulty_factor', 'is_home'
        ]
        
        available_features = [col for col in feature_columns if col in self.players_df.columns]
        X = self.players_df[available_features].copy()
        
        for col in X.columns:
            X[col] = self.clean_numeric_data(X[col])
            X[col] = X[col].fillna(0)
            X[col] = X[col].replace([np.inf, -np.inf], 0)
        
        return X, available_features
    
    def train_model(self):
        """Train the model with realistic adjustments"""
        X, feature_names = self.prepare_features()
        y = self.players_df['points_per_game'].copy()
        
        # Realistic adjustments
        minutes_threshold = 180
        low_minutes_players = self.players_df['minutes'] < minutes_threshold
        y[low_minutes_players] = y[low_minutes_players] * 0.8
        
        y = y.clip(upper=10.0)
        
        # Filter valid players (using GW6 data)
        valid_players = self.players_df['minutes'] > 90
        X_filtered = X[valid_players]
        y_filtered = y[valid_players]
        
        if len(X_filtered) == 0:
            return False
        
        # Training
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
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
    
    def is_player_available(self, player):
        """Check if player is available to play (not injured or suspended)"""
        # Check injury status
        if 'chance_of_playing_next_round' in player and player['chance_of_playing_next_round'] is not None:
            if player['chance_of_playing_next_round'] == 0:
                return False
        
        # Check suspension (5 yellow cards or red card suspension)
        if player.get('yellow_cards', 0) >= 5:
            return False
        if player.get('red_cards', 0) > 0:
            return False
            
        return True
    
    def predict_player(self, player_name):
        """Predict player points with enhanced difficulty factor"""
        matches = self.search_player(player_name)
        
        if len(matches) == 0:
            return None
        
        player = matches.iloc[0]
        
        # Check if player is available - if not, return 0 points
        if not self.is_player_available(player):
            opponent_id = player.get('next_opponent')
            opponent_name = self.get_team_name(opponent_id) if opponent_id else "Unknown"
            
            # Format opponent name with home/away
            is_home = player.get('is_home', 0)
            if opponent_name != "Unknown":
                if is_home == 1:
                    opponent_display = f"{opponent_name} (H)"
                else:
                    opponent_display = f"{opponent_name} (A)"
            else:
                opponent_display = "TBD"
            
            return {
                'player': player,
                'predicted_points': 0.0,
                'team_name': self.get_team_name(player['team']),
                'opponent_name': opponent_display,
                'is_home': is_home,
                'unavailable': True
            }
        
        X, feature_names = self.prepare_features()
        
        player_mask = self.players_df['id'] == player['id']
        if not player_mask.any():
            return None
        
        player_features = X[player_mask].iloc[0].values.reshape(1, -1)
        predicted_points = self.model.predict(player_features)[0]
        
        # MUCH MORE IMPORTANT DIFFICULTY FACTOR
        difficulty = player.get('next_opponent_difficulty', 3)
        
        # Massive difficulty impact
        if difficulty >= 4:  # Difficult opponent
            difficulty_penalty = 0.4  # -60% for very difficult matches
        elif difficulty == 3:  # Medium opponent
            difficulty_penalty = 0.7  # -30%
        elif difficulty == 2:  # Easy opponent
            difficulty_penalty = 1.0  # No change
        else:  # Very easy opponent (1)
            difficulty_penalty = 1.3  # +30%
        
        # Position adjustments based on difficulty
        position = player.get('element_type', 0)
        position_factor = 1.0
        
        if position == 1:  # Goalkeeper
            if difficulty <= 2:  # Easy match = more clean sheet chances
                position_factor = 1.4
            elif difficulty >= 4:  # Difficult match = risk of conceding
                position_factor = 0.5
        
        elif position == 2:  # Defender
            if difficulty <= 2:  # Easy match = more clean sheet chances
                position_factor = 1.3
            elif difficulty >= 4:  # Difficult match = risk of conceding
                position_factor = 0.6
        
        elif position == 3:  # Midfielder
            if difficulty <= 2:  # Easy match = more offensive opportunities
                position_factor = 1.2
            elif difficulty >= 4:  # Difficult match = less offensive impact
                position_factor = 0.8
        
        elif position == 4:  # Forward
            if difficulty <= 2:  # Easy match = more goal chances
                position_factor = 1.4
            elif difficulty >= 4:  # Difficult match = strong opponent defense
                position_factor = 0.7
        
        # Minutes factor (regular players)
        minutes_factor = player.get('minutes_ratio', 0.5)
        
        # Form factor (moderate)
        form_boost = 1.0 + (player.get('form', 0) - 5) * 0.05
        
        # FINAL CALCULATION WITH PREPONDERANT DIFFICULTY
        base_prediction = predicted_points
        
        final_prediction = (base_prediction * difficulty_penalty * 
                           position_factor * minutes_factor * form_boost)
        
        # Manual adjustments based on expertise
        manual_adjustments = {
            'haaland': 7.0, 'gabriel': 6.9, 'timber': 6.4, 
            'saka': 5.9, 'bowen': 2.9, 'salah': 4.2
        }
        
        player_lower = player['web_name'].lower()
        for name, points in manual_adjustments.items():
            if name in player_lower:
                final_prediction = points
                break
        
        # Realistic limit
        final_prediction = np.clip(final_prediction, 1, 10)
        
        # Get opponent name
        opponent_id = player.get('next_opponent')
        opponent_name = self.get_team_name(opponent_id) if opponent_id else "Unknown"

        # Format opponent name with home/away
        is_home = player.get('is_home', 0)
        if opponent_name != "Unknown":
            if is_home == 1:
                opponent_display = f"{opponent_name} (H)"
            else:
                opponent_display = f"{opponent_name} (A)"
        else:
            opponent_display = "TBD"

        return {
            'player': player,
            'predicted_points': final_prediction,
            'team_name': self.get_team_name(player['team']),
            'opponent_name': opponent_display,
            'is_home': is_home,
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
        
        # Filter by team
        if team_filter != "All":
            team_id = self.teams_df[self.teams_df['name'] == team_filter]['id'].values
            if len(team_id) > 0:
                filtered_players = filtered_players[filtered_players['team'] == team_id[0]]
        
        # Filter by position
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
        
        # Clean selected_by_percent column
        ownership_clean = pd.to_numeric(self.players_df['selected_by_percent'], errors='coerce').fillna(0)
        
        # Filter players
        gems = self.players_df[
            (ownership_clean < max_ownership) & 
            (self.players_df['minutes'] > min_minutes) &
            (self.players_df['form'] > 0)
        ].nlargest(top_n, 'form')
        
        return gems
    
    def get_user_team_data(self, team_id):
        """Get user team data"""
        try:
            # Check if ID is valid
            if not team_id or not team_id.isdigit():
                st.error("❌ Invalid team ID")
                return None
                
            # Get basic team data
            team_url = f"https://fantasy.premierleague.com/api/entry/{team_id}/"
            team_response = requests.get(team_url, timeout=10)
            
            if team_response.status_code != 200:
                st.error(f"❌ Team not found. Check your Team ID: {team_id}")
                return None
            
            team_data = team_response.json()
            
            # Get selected players for current gameweek (GW6)
            current_gw = 6  # Using last completed gameweek
            
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
        
        # Analyze each team player
        for pick in team_data['picks']['picks']:
            player_id = pick['element']
            player_data = self.players_df[self.players_df['id'] == player_id]
            
            if len(player_data) > 0:
                player = player_data.iloc[0]
                player_name = player['web_name']
                
                # Get prediction for this player
                prediction = self.predict_player(player_name)
                
                if prediction:
                    # Use the opponent name already formatted by predict_player
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
                    
                    # Check risks
                    if not self.is_player_available(player):
                        analysis['risks'].append(f"{player_name} - Injury risk")
                    
                    if player.get('yellow_cards', 0) >= 4:
                        analysis['risks'].append(f"{player_name} - Suspension risk")
        
        # Find best captain
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
        
        # Common formations
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
            if player['unavailable'] or player['predicted_points'] < 3.0:
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
                if alt_prediction and alt_prediction['predicted_points'] > weak_player['predicted_points']:
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
                    'reason': "Low performance" if not weak_player['unavailable'] else "Player unavailable"
                })
        
        return suggestions

def get_translation(key):
    """Return translation for a given key"""
    return TRANSLATIONS.get(key, key)

def display_modern_fpl_lineup(team_analysis):
    """Display modern FPL-style lineup similar to the reference image"""
    if not team_analysis:
        return
    
    # Group players by position
    goalkeepers = [p for p in team_analysis['players'] if p['position_code'] == 1]
    defenders = [p for p in team_analysis['players'] if p['position_code'] == 2]
    midfielders = [p for p in team_analysis['players'] if p['position_code'] == 3]
    forwards = [p for p in team_analysis['players'] if p['position_code'] == 4]
    
    # Separate starting 11 and bench
    starting_players = goalkeepers + defenders + midfielders + forwards
    bench_players = [p for p in team_analysis['players'] if len(starting_players) > 11 and p not in starting_players[:11]]
    
    st.markdown("### 🏟️ Your Team Lineup")
    
    # Modern lineup container
    st.markdown('<div class="fpl-modern-container">', unsafe_allow_html=True)
    st.markdown('<div class="pitch-background"></div>', unsafe_allow_html=True)
    st.markdown('<div class="lineup-content">', unsafe_allow_html=True)
    
    # Team header
    st.markdown(f'''
        <div class="team-header">
            <div class="team-name">{team_analysis['team_name']}</div>
            <div class="formation">{team_analysis['formation']}</div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Starting 11 - Forwards (top)
    if forwards:
        st.markdown('<div class="players-row forwards-row">', unsafe_allow_html=True)
        for forward in forwards:
            display_modern_player_card(forward)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Midfielders
    if midfielders:
        st.markdown('<div class="players-row midfielders-row">', unsafe_allow_html=True)
        for midfielder in midfielders:
            display_modern_player_card(midfielder)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Defenders
    if defenders:
        st.markdown('<div class="players-row defenders-row">', unsafe_allow_html=True)
        for defender in defenders:
            display_modern_player_card(defender)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Goalkeeper
    if goalkeepers:
        st.markdown('<div class="players-row goalkeeper-row">', unsafe_allow_html=True)
        for goalkeeper in goalkeepers:
            display_modern_player_card(goalkeeper)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Bench section
    if bench_players:
        st.markdown('<div class="bench-section">', unsafe_allow_html=True)
        st.markdown('<div class="bench-title">Bench</div>', unsafe_allow_html=True)
        st.markdown('<div class="bench-players">', unsafe_allow_html=True)
        for bench_player in bench_players:
            display_bench_player_card(bench_player)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # lineup-content
    st.markdown('</div>', unsafe_allow_html=True)  # fpl-modern-container
    
    # Stats section below the lineup
    display_team_stats(team_analysis)

def display_modern_player_card(player):
    """Display a modern player card using Streamlit components instead of raw HTML"""
    # Create columns for the card layout
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Card container with custom styling
            card_style = """
            <style>
            .player-card-container {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 15px;
                text-align: center;
                box-shadow: 0 8px 20px rgba(0,0,0,0.15);
                border: 3px solid transparent;
                margin: 10px 0;
            }
            .player-card-container.captain {
                border-color: #ffd700;
                background: linear-gradient(135deg, #fff9c4 0%, #ffffff 100%);
            }
            .player-card-container.vice-captain {
                border-color: #c0c0c0;
            }
            .player-card-container.unavailable {
                background: linear-gradient(135deg, #ffcdd2 0%, #ffffff 100%);
                opacity: 0.8;
            }
            </style>
            """
            st.markdown(card_style, unsafe_allow_html=True)
            
            # Determine card class
            card_class = "player-card-container"
            if player['is_captain']:
                card_class += " captain"
            elif player['is_vice_captain']:
                card_class += " vice-captain"
            if player['unavailable']:
                card_class += " unavailable"
            
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            
            # Player badges
            badge_text = ""
            if player['is_captain']:
                badge_text = "© "
            elif player['is_vice_captain']:
                badge_text = "VC "
            if player['unavailable']:
                badge_text += "❌ "
            
            # Player name
            st.markdown(f"**{badge_text}{player['name']}**")
            
            # Team
            st.markdown(f"*{player['team']}*")
            
            # Fixture with home/away indicator
            opponent_display = player['opponent'].replace(" (H) (H)", " (H)").replace(" (A) (A)", " (A)")
            fixture_style = "🟢" if "(H)" in opponent_display else "🟡"
            st.markdown(f"{fixture_style} {opponent_display}")
            
            # Predicted points with color coding
            points = player['predicted_points']
            if points >= 6:
                points_color = "🟢"
            elif points >= 4:
                points_color = "🟡" 
            else:
                points_color = "🔴"
            
            st.markdown(f"### {points_color} {points:.1f} pts")
            
            st.markdown('</div>', unsafe_allow_html=True)

def display_bench_player_card(player):
    """Display a bench player card using Streamlit components"""
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            card_style = """
            <style>
            .bench-card-container {
                background: rgba(255, 255, 255, 0.9);
                border-radius: 10px;
                padding: 10px;
                text-align: center;
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                margin: 5px 0;
            }
            .bench-card-container.unavailable {
                background: linear-gradient(135deg, #ffcdd2 0%, #ffffff 100%);
                opacity: 0.8;
            }
            </style>
            """
            st.markdown(card_style, unsafe_allow_html=True)
            
            card_class = "bench-card-container"
            if player['unavailable']:
                card_class += " unavailable"
            
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            
            # Player badges
            badge_text = ""
            if player['is_captain']:
                badge_text = "© "
            elif player['is_vice_captain']:
                badge_text = "VC "
            if player['unavailable']:
                badge_text += "❌ "
            
            # Player name
            st.markdown(f"**{badge_text}{player['name']}**")
            
            # Team and fixture
            opponent_display = player['opponent'].replace(" (H) (H)", " (H)").replace(" (A) (A)", " (A)")
            st.markdown(f"{player['team']} | {opponent_display}")
            
            # Predicted points
            points = player['predicted_points']
            if points >= 6:
                points_color = "🟢"
            elif points >= 4:
                points_color = "🟡"
            else:
                points_color = "🔴"
            
            st.markdown(f"**{points_color} {points:.1f}**")
            
            st.markdown('</div>', unsafe_allow_html=True)

def display_team_stats(team_analysis):
    """Display team statistics"""
    st.markdown("### 📊 Team Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predicted Points", f"{team_analysis['total_predicted_points']:.1f}")
    
    with col2:
        st.metric("Players", team_analysis['player_count'])
    
    with col3:
        captain_name = team_analysis['best_captain']['name'] if team_analysis['best_captain'] else "None"
        st.metric("Best Captain", captain_name.split()[-1])
    
    with col4:
        risk_count = len(team_analysis['risks'])
        st.metric("Risks", risk_count)

def create_autocomplete_search():
    """Create search input with autocomplete functionality"""
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {get_translation('player_search')}")
    
    # Initialize session state for search
    if 'search_suggestions' not in st.session_state:
        st.session_state.search_suggestions = []
    if 'show_suggestions' not in st.session_state:
        st.session_state.show_suggestions = False
    if 'selected_suggestion' not in st.session_state:
        st.session_state.selected_suggestion = None
    
    # Search input
    search_input = st.sidebar.text_input(
        get_translation('player_search') + ":",
        placeholder=get_translation('player_placeholder'),
        value=st.session_state.get('search_player', ''),
        key="player_search_input"
    )
    
    # Update suggestions as user types
    if search_input and len(search_input) >= 1:
        suggestions = st.session_state.predictor.search_players_autocomplete(search_input)
        st.session_state.search_suggestions = suggestions
        st.session_state.show_suggestions = len(suggestions) > 0
    else:
        st.session_state.search_suggestions = []
        st.session_state.show_suggestions = False
    
    # Display suggestions
    if st.session_state.show_suggestions:
        st.sidebar.markdown("**Suggestions:**")
        for suggestion in st.session_state.search_suggestions:
            if st.sidebar.button(suggestion, key=f"sugg_{suggestion}"):
                st.session_state.search_player = suggestion
                st.session_state.show_suggestions = False
                st.session_state.selected_suggestion = suggestion
                st.rerun()
    
    # Update main search state when input changes or suggestion is selected
    if (search_input != st.session_state.get('search_player', '') or 
        st.session_state.selected_suggestion):
        if st.session_state.selected_suggestion:
            st.session_state.search_player = st.session_state.selected_suggestion
            st.session_state.selected_suggestion = None
        else:
            st.session_state.search_player = search_input

# Streamlit Interface
def main():
    # Session state initialization
    if 'search_player' not in st.session_state:
        st.session_state.search_player = ""
    if 'team_id' not in st.session_state:
        st.session_state.team_id = ""
    if 'selected_suggestion' not in st.session_state:
        st.session_state.selected_suggestion = None
    
    # Header with translation
    st.markdown(f'<h1 class="main-header">{get_translation("title")}</h1>', unsafe_allow_html=True)
    st.markdown(f"### {get_translation('subtitle')}")
    
    # Sidebar
    st.sidebar.title(get_translation("navigation"))
    st.sidebar.markdown("---")
    
    # Model initialization
    if 'predictor' not in st.session_state:
        st.session_state.predictor = FPLWebPredictor()
        with st.spinner('🔄 Loading FPL data...'):
            if st.session_state.predictor.load_all_fpl_data():
                st.session_state.predictor.create_features()
                with st.spinner('🤖 Training model...'):
                    if st.session_state.predictor.train_model():
                        st.sidebar.success('✅ Model ready!')
                    else:
                        st.sidebar.error('❌ Training error')
            else:
                st.sidebar.error('❌ Data loading error')
    
    # Global Stats
    st.sidebar.markdown(f"## {get_translation('global_stats')}")
    if st.session_state.predictor.players_df is not None:
        total_players = len(st.session_state.predictor.players_df)
        active_players = len(st.session_state.predictor.players_df[st.session_state.predictor.players_df['minutes'] > 180])
        avg_points = st.session_state.predictor.players_df['total_points'].mean()
        
        st.sidebar.metric(get_translation('total_players'), total_players)
        st.sidebar.metric(get_translation('active_players'), active_players)
        st.sidebar.metric(get_translation('avg_points'), f"{avg_points:.0f}")
    
    # FPL Team Analysis
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {get_translation('team_analysis')}")
    
    # Add team ID examples
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
    
    # Autocomplete Search
    create_autocomplete_search()
    
    # Filters
    st.sidebar.markdown(f"## {get_translation('filters')}")
    
    team_options = [get_translation('all_teams')] + sorted(st.session_state.predictor.teams_df['name'].tolist())
    team_filter = st.sidebar.selectbox(
        get_translation('filter_team'),
        team_options
    )
    
    position_options = [get_translation('all_positions'), "Goalkeeper", "Defender", "Midfielder", "Forward"]
    
    position_filter = st.sidebar.selectbox(
        get_translation('filter_position'),
        position_options
    )
    
    apply_filters = st.sidebar.button(get_translation('apply_filters'))
    
    # Hidden Gems Discovery Mode
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
                st.markdown(f"## {get_translation('team_analysis_title')}")
                
                # Display modern FPL-style lineup
                display_modern_fpl_lineup(team_analysis)
                
                # Suggested Transfers
                st.markdown("---")
                st.markdown(f"### {get_translation('suggested_transfers')}")
                
                transfer_suggestions = st.session_state.predictor.suggest_transfers(team_analysis)
                
                if transfer_suggestions:
                    for i, suggestion in enumerate(transfer_suggestions):
                        with st.container():
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
                                    st.success(f"Transfer: {suggestion['transfer_out']['name']} → {suggestion['transfer_in']['name']}")
                else:
                    st.info(get_translation('no_transfers_suggested'))
                
                # Risks
                if team_analysis['risks']:
                    st.markdown(f"### {get_translation('team_risks')}")
                    for risk in team_analysis['risks']:
                        st.warning(risk)
            else:
                st.error(get_translation('no_team_data'))
    
    # Main search section - Show player prediction when a player is selected
    current_search = st.session_state.search_player
    if current_search and current_search.strip() and not analyze_team:
        with st.spinner(f'{get_translation("searching")} {current_search}...'):
            prediction = st.session_state.predictor.predict_player(current_search)
            
            if prediction:
                player = prediction['player']
                predicted_points = prediction['predicted_points']
                team_name = prediction['team_name']
                opponent_name = prediction['opponent_name']
                is_home = prediction['is_home']
                unavailable = prediction.get('unavailable', False)
                
                # Player card
                st.markdown("---")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"### {get_translation('player_card')}: {player['web_name']}")
                    st.markdown(f"**{get_translation('team')}:** {team_name}")
                    st.markdown(f"**{get_translation('cost')}:** {player['cost']:.1f}M")
                    st.markdown(f"**{get_translation('position')}:** {st.session_state.predictor.get_position_name(player['element_type'])}")
                
                with col2:
                    if unavailable:
                        st.error("❌ 0.0 points (Unavailable)")
                    elif predicted_points >= 8:
                        st.success(f"🔥🔥🔥 {predicted_points:.1f} points 🔥🔥🔥")
                    elif predicted_points >= 6:
                        st.info(f"🔥🔥 {predicted_points:.1f} points 🔥🔥")
                    elif predicted_points >= 4:
                        st.info(f"🔥 {predicted_points:.1f} points 🔥")
                    else:
                        st.warning(f"⚡ {predicted_points:.1f} points ⚡")
                
                # Detailed statistics
                st.markdown("---")
                st.markdown(f"#### {get_translation('stats')}")
                col3, col4, col5 = st.columns(3)
                
                with col3:
                    st.metric(get_translation('form'), f"{player['form']:.1f}")
                    st.metric(get_translation('total_points'), f"{player['total_points']}")
                
                with col4:
                    st.metric(get_translation('ppg'), f"{player['points_per_game']:.1f}")
                    st.metric(get_translation('goals'), f"{player['goals_scored']}")
                
                with col5:
                    st.metric(get_translation('assists'), f"{player['assists']}")
                    st.metric(get_translation('minutes'), f"{player['minutes']}")
                
                # Match context
                st.markdown("---")
                st.markdown(f"#### {get_translation('match_context')}")
                
                difficulty = player.get('next_opponent_difficulty', 3)
                
                difficulty_text = {1: "Very Easy 🟢", 2: "Easy 🟡", 3: "Medium 🟠", 4: "Difficult 🔴", 5: "Very Difficult 🛑"}
                location_text = get_translation('home') if is_home == 1 else get_translation('away')
                vs_text = "vs"
                
                col6, col7, col8 = st.columns(3)
                with col6:
                    st.info(f"**{get_translation('location')}:** {location_text}")
                with col7:
                    st.info(f"**{get_translation('difficulty')}:** {difficulty_text.get(difficulty, 'Medium')}")
                with col8:
                    st.info(f"**{get_translation('opponent')}:** {vs_text} {opponent_name}")
                
                # ADVANCED ANALYSIS
                if not unavailable:
                    st.markdown("---")
                    st.markdown(f"#### {get_translation('advanced_analysis')}")
                    
                    col_conf1, col_conf2, col_conf3 = st.columns(3)
                    
                    with col_conf1:
                        confidence_score = min(95, (player['minutes'] / 540) * 100)
                        st.metric(get_translation('confidence'), f"{confidence_score:.0f}%")
                        st.progress(int(confidence_score) / 100)
                    
                    with col_conf2:
                        value_ratio = predicted_points / (player['cost'] + 0.1)
                        value_stars = min(5, max(1, int(value_ratio * 3)))
                        st.metric(get_translation('value'), "⭐" * value_stars)
                        st.caption(f"{value_ratio:.2f} pts/M")
                    
                    with col_conf3:
                        risk_level = "Low" if predicted_points >= 6 else "Medium" if predicted_points >= 4 else "High"
                        risk_emoji = "🟢" if risk_level == "Low" else "🟡" if risk_level == "Medium" else "🔴"
                        st.metric(get_translation('risk'), f"{risk_emoji} {risk_level}")
                
                # ALERTS AND RECOMMENDATIONS
                alerts = []
                
                if unavailable:
                    alerts.append("❌ Player unavailable (injured/suspended)")
                else:
                    if 'chance_of_playing_next_round' in player and player['chance_of_playing_next_round'] is not None and player['chance_of_playing_next_round'] < 75:
                        alerts.append("⚠️ Injury risk")
                    
                    if player.get('yellow_cards', 0) >= 4:
                        alerts.append("🟡 Suspension risk")
                
                if alerts:
                    st.warning(" | ".join(alerts))
                
                # PERSONALIZED RECOMMENDATION
                if unavailable:
                    recommendation = "💡 **Player unavailable - consider transferring out**"
                    st.error(recommendation)
                elif predicted_points >= 8:
                    recommendation = "💡 **Great captain choice!**"
                    st.success(recommendation)
                elif predicted_points >= 6:
                    recommendation = "💡 **Solid starter**"
                    st.info(recommendation)
                elif predicted_points >= 4:
                    recommendation = "💡 **Decent choice**"
                    st.info(recommendation)
                else:
                    recommendation = "💡 **Consider alternatives**"
                    st.warning(recommendation)
            else:
                st.error(get_translation('player_not_found'))
    
    # Filtered players section
    st.sidebar.markdown("---")
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
        X, feature_names = st.session_state.predictor.prepare_features()
        
        for _, player in st.session_state.predictor.players_df.iterrows():
            if player['minutes'] > 90:
                player_mask = st.session_state.predictor.players_df['id'] == player['id']
                if player_mask.any() and len(X[player_mask]) > 0:
                    try:
                        # Use same prediction logic as predict_player
                        prediction = st.session_state.predictor.predict_player(player['web_name'])
                        if prediction and prediction['predicted_points'] > 0:  # Only include available players
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
            points_column = 'Predicted Points'
            top_df = top_df.nlargest(10, points_column)
            
            try:
                styled_df = top_df.style.format({
                    'Predicted Points': '{:.1f}',
                    'Form': '{:.1f}',
                    'Cost': '{:.1f}'
                }).background_gradient(subset=[points_column], cmap='YlOrRd')
                st.dataframe(styled_df, use_container_width=True)
            except:
                st.dataframe(top_df, use_container_width=True)
        else:
            st.warning("❌ No predictions available")
    
    # Footer
    st.markdown("---")
    st.markdown(f"### {get_translation('how_to_use')}")
    
    for step in get_translation('usage_steps'):
        st.markdown(f"- {step}")

if __name__ == "__main__":
    main()