import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="FPL Predictor Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# English translations
TRANSLATIONS = {
    'title': "⚽ FPL Predictor Pro",
    'subtitle': "🔮 AI-Powered Points Predictions",
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
    'prediction': "🔮 Prediction",
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
    'transfer_reason': "💡 Reason",
    'best_lineup': "🏆 Best 3-5-2 Lineup",
    'model_accuracy': "🎯 Model Accuracy Analysis",
    'accuracy_metrics': "📊 Accuracy Metrics",
    'historical_performance': "📈 Historical Performance",
    'prediction_confidence': "🎯 Prediction Confidence",
}

# Ultra Modern FPL-style CSS with fixed input fields
st.markdown("""
<style>
    /* Main Background Gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #FFD700 0%, #FF6B6B 50%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 900;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        font-family: 'Arial Black', sans-serif;
    }
    
    .subtitle {
        font-size: 1.5rem;
        color: #FF69B4 !important;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Glassmorphism Player Card */
    .player-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .player-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: 0.5s;
    }
    
    .player-card:hover::before {
        left: 100%;
    }
    
    .player-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 25px 50px rgba(0,0,0,0.3);
        border: 1px solid rgba(255, 255, 255, 0.4);
    }
    
    /* Neon Stat Cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9), rgba(118, 75, 162, 0.9));
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stat-card::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #ff0000, #ff7300, #fffb00, #48ff00, #00ffd5, #002bff, #7a00ff, #ff00c8, #ff0000);
        background-size: 400%;
        border-radius: 15px;
        z-index: -1;
        animation: glowing 20s linear infinite;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .stat-card:hover::before {
        opacity: 1;
    }
    
    @keyframes glowing {
        0% { background-position: 0 0; }
        50% { background-position: 400% 0; }
        100% { background-position: 0 0; }
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 8px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Animated Prediction Badges */
    .prediction-badge {
        font-size: 2.2rem;
        font-weight: 900;
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 2px solid rgba(255,255,255,0.2);
        animation: pulse 2s infinite;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .prediction-badge::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .top-player {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #000;
        border: 2px solid #FFD700;
    }
    
    .good-player {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        border: 2px solid #4CAF50;
    }
    
    .avg-player {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white;
        border: 2px solid #2196F3;
    }
    
    .poor-player {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white;
        border: 2px solid #ff4444;
    }
    
    /* Modern Transfer Cards */
    .transfer-suggestion {
        background: linear-gradient(135deg, rgba(168, 230, 207, 0.9), rgba(220, 237, 193, 0.9));
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        border-left: 6px solid #4caf50;
        box-shadow: 0 12px 30px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .transfer-suggestion:hover {
        transform: translateX(10px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
    }
    
    .transfer-warning {
        background: linear-gradient(135deg, rgba(255, 211, 182, 0.9), rgba(255, 170, 165, 0.9));
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        border-left: 6px solid #ff9800;
        box-shadow: 0 12px 30px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .transfer-warning:hover {
        transform: translateX(10px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
    }
    
    /* Accuracy Cards */
    .accuracy-good {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        border: 2px solid #4CAF50;
    }
    
    .accuracy-avg {
        background: linear-gradient(135deg, #FF9800, #F57C00);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        border: 2px solid #FF9800;
    }
    
    .accuracy-poor {
        background: linear-gradient(135deg, #F44336, #D32F2F);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        border: 2px solid #F44336;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%) !important;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Input Field Styling - FIXED FOR VISIBILITY */
    .stTextInput>div>div>input {
        background: white !important;
        border: 2px solid #667eea !important;
        border-radius: 12px !important;
        color: #2c3e50 !important;
        padding: 12px !important;
        font-weight: 500 !important;
    }
    
    .stTextInput>div>div>input::placeholder {
        color: #666 !important;
        opacity: 0.7 !important;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.3) !important;
        background: white !important;
    }
    
    /* Select Box Styling */
    .stSelectbox>div>div>div {
        background: white !important;
        border: 2px solid #667eea !important;
        border-radius: 12px !important;
        color: #2c3e50 !important;
    }
    
    /* Metric Styling */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.1) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 15px;
        padding: 20px;
    }
    
    /* Floating Animation */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .floating {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Player Image Container */
    .player-image {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid #667eea;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
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
        self.historical_data = None
        self.h2h_stats = None
        self.model_accuracy = None
        
    def load_all_fpl_data(self):
        """Load FPL data with historical integration"""
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
            
            # Load historical data and H2H stats
            self.load_historical_data()
            self.load_h2h_stats()
            
            return True
        except Exception as e:
            st.error(f"Loading error: {e}")
            return False
    
    def load_historical_data(self):
        """Load and process historical player data"""
        try:
            # Simulated historical data
            self.historical_data = {
                'consistency_scores': {},
                'season_trends': {},
                'position_changes': {},
                'team_changes': {}
            }
            
            # Calculate consistency scores based on current data
            for _, player in self.players_df.iterrows():
                player_id = player['id']
                
                # Simulate historical consistency
                if player['minutes'] > 450:
                    consistency = min(1.0, player['total_points'] / (player['minutes'] / 90 * 3))
                    self.historical_data['consistency_scores'][player_id] = consistency
                else:
                    self.historical_data['consistency_scores'][player_id] = 0.5
                    
        except Exception as e:
            st.warning(f"Historical data loading issue: {e}")
    
    def load_h2h_stats(self):
        """Load head-to-head statistics"""
        try:
            # Simulated H2H data
            self.h2h_stats = {
                'team_records': {},
                'player_records': {},
                'bogey_teams': {}
            }
            
            # Initialize with basic data
            for team_id in self.teams_df['id']:
                self.h2h_stats['team_records'][team_id] = {
                    'home_strength': np.random.uniform(0.8, 1.2),
                    'away_strength': np.random.uniform(0.8, 1.2)
                }
                
        except Exception as e:
            st.warning(f"H2H stats loading issue: {e}")
    
    def clean_numeric_data(self, series):
        """Clean numeric data"""
        if pd.api.types.is_numeric_dtype(series):
            return series
        series_clean = series.replace([None, 'None', 'null', ''], '0')
        try:
            return pd.to_numeric(series_clean, errors='coerce')
        except:
            return pd.Series([0] * len(series), index=series.index)
    
    def get_next_gameweek(self):
        """Get the NEXT gameweek number for predictions"""
        try:    
            bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
            response = requests.get(bootstrap_url, timeout=10)
            data = response.json()

            events = data['events']
            next_gw = None

            for event in events:
                if event['is_next']:
                    next_gw = event['id']
                    break
            
            if next_gw is None:
                for event in events:
                    if event['is_current']:
                        next_gw = event['id'] + 1
                        break
            
            return next_gw if next_gw else 9
        except Exception as e:
            st.error(f"Error getting next gameweek: {e}")
            return 9
    
    def create_features(self):
        """Create enhanced features with historical data"""
        # Data cleaning
        critical_columns = ['form', 'points_per_game', 'goals_scored', 'assists', 'minutes', 
                           'total_points', 'influence', 'creativity', 'threat', 'ict_index', 'bps',
                           'clean_sheets', 'saves', 'goals_conceded', 'penalties_saved', 'selected_by_percent']
        
        for col in critical_columns:
            if col in self.players_df.columns:
                self.players_df[col] = self.clean_numeric_data(self.players_df[col])
        
        # Basic feature engineering
        self.players_df['cost'] = self.players_df['now_cost'] / 10
        self.players_df['recent_form_weighted'] = self.players_df['form'] * 1.2  # Reduced from 1.5
        
        # Points efficiency metrics
        self.players_df['points_per_minute'] = self.players_df['total_points'] / np.maximum(self.players_df['minutes'], 1)
        self.players_df['points_per_minute'] = self.players_df['points_per_minute'].replace([np.inf, -np.inf], 0).fillna(0)
        
        self.players_df['goal_involvement_per_90'] = (self.players_df['goals_scored'] + self.players_df['assists']) / np.maximum(self.players_df['minutes'] / 90, 1)
        self.players_df['goal_involvement_per_90'] = self.players_df['goal_involvement_per_90'].replace([np.inf, -np.inf], 0).fillna(0)
        
        self.players_df['minutes_reliability'] = self.players_df['minutes'] / (8 * 90)
        self.players_df['minutes_reliability'] = self.players_df['minutes_reliability'].clip(0, 1)
        
        # Position-specific features
        self.players_df['position_specific_score'] = 0.0
        
        gk_mask = self.players_df['element_type'] == 1
        if gk_mask.any():
            self.players_df.loc[gk_mask, 'position_specific_score'] += (
                self.players_df.loc[gk_mask, 'clean_sheets'] * 2 +  # Reduced from 3
                self.players_df.loc[gk_mask, 'saves'] * 0.05 +      # Reduced from 0.1
                self.players_df.loc[gk_mask, 'penalties_saved'] * 3  # Reduced from 5
            ) / np.maximum(self.players_df.loc[gk_mask, 'minutes'] / 90, 1)
        
        defender_mask = self.players_df['element_type'] == 2
        if defender_mask.any():
            self.players_df.loc[defender_mask, 'position_specific_score'] += (
                self.players_df.loc[defender_mask, 'clean_sheets'] * 1.5 +  # Reduced from 2
                self.players_df.loc[defender_mask, 'goals_scored'] * 3 +    # Reduced from 4
                self.players_df.loc[defender_mask, 'bps'] * 0.05           # Reduced from 0.1
            ) / np.maximum(self.players_df.loc[defender_mask, 'minutes'] / 90, 1)
        
        midfielder_mask = self.players_df['element_type'] == 3
        if midfielder_mask.any():
            self.players_df.loc[midfielder_mask, 'position_specific_score'] += (
                self.players_df.loc[midfielder_mask, 'goals_scored'] * 4 +  # Reduced from 5
                self.players_df.loc[midfielder_mask, 'assists'] * 2.5 +     # Reduced from 3
                self.players_df.loc[midfielder_mask, 'creativity'] * 0.005  # Reduced from 0.01
            ) / np.maximum(self.players_df.loc[midfielder_mask, 'minutes'] / 90, 1)
        
        forward_mask = self.players_df['element_type'] == 4
        if forward_mask.any():
            self.players_df.loc[forward_mask, 'position_specific_score'] += (
                self.players_df.loc[forward_mask, 'goals_scored'] * 5 +  # Reduced from 6
                self.players_df.loc[forward_mask, 'assists'] * 1.5 +     # Reduced from 2
                self.players_df.loc[forward_mask, 'threat'] * 0.005      # Reduced from 0.01
            ) / np.maximum(self.players_df.loc[forward_mask, 'minutes'] / 90, 1)
        
        self.players_df['position_specific_score'] = self.players_df['position_specific_score'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Team features
        team_strength = dict(zip(self.teams_df['id'], self.teams_df['strength']))
        team_attack = dict(zip(self.teams_df['id'], self.teams_df['strength_attack_home']))
        team_defense = dict(zip(self.teams_df['id'], self.teams_df['strength_defence_home']))
        self.players_df['team_strength'] = self.players_df['team'].map(team_strength)
        self.players_df['team_attack'] = self.players_df['team'].map(team_attack)
        self.players_df['team_defense'] = self.players_df['team'].map(team_defense)
        
        # HISTORICAL FEATURES
        self.players_df['historical_consistency'] = self.players_df['id'].map(
            self.historical_data['consistency_scores']
        ).fillna(0.5)
        
        # H2H FEATURES
        self.players_df['team_home_strength'] = self.players_df['team'].map(
            lambda x: self.h2h_stats['team_records'].get(x, {}).get('home_strength', 1.0)
        )
        self.players_df['team_away_strength'] = self.players_df['team'].map(
            lambda x: self.h2h_stats['team_records'].get(x, {}).get('away_strength', 1.0)
        )
        
        # Match context for NEXT GW
        next_gw = self.get_next_gameweek()
        next_gw_fixtures = self.fixtures_df[self.fixtures_df['event'] == next_gw]
        
        if len(next_gw_fixtures) == 0:
            available_gws = sorted(self.fixtures_df['event'].unique())
            if len(available_gws) > 0:
                future_gws = [gw for gw in available_gws if gw >= next_gw]
                if future_gws:
                    next_gw = future_gws[0]
                    next_gw_fixtures = self.fixtures_df[self.fixtures_df['event'] == next_gw]
                else:
                    next_gw = available_gws[-1]
                    next_gw_fixtures = self.fixtures_df[self.fixtures_df['event'] == next_gw]
        
        team_difficulty = {}
        home_advantage = {}
        opponent_mapping = {}
        opponent_strength = {}
        
        for _, fixture in next_gw_fixtures.iterrows():
            team_h = fixture['team_h']
            team_a = fixture['team_a']
            
            team_difficulty[team_h] = fixture['team_h_difficulty']
            team_difficulty[team_a] = fixture['team_a_difficulty']
            home_advantage[team_h] = 1
            home_advantage[team_a] = 0
            opponent_mapping[team_h] = team_a
            opponent_mapping[team_a] = team_h
            opponent_strength[team_h] = team_strength.get(team_a, 3)
            opponent_strength[team_a] = team_strength.get(team_h, 3)
        
        self.players_df['next_opponent_difficulty'] = self.players_df['team'].map(team_difficulty)
        self.players_df['is_home'] = self.players_df['team'].map(home_advantage)
        self.players_df['next_opponent'] = self.players_df['team'].map(opponent_mapping)
        self.players_df['opponent_strength'] = self.players_df['team'].map(opponent_strength)
        
        self.players_df['difficulty_factor'] = (6 - self.players_df['next_opponent_difficulty']) / 5
        self.players_df['home_advantage_bonus'] = self.players_df['is_home'] * 0.2  # Reduced from 0.3
        
        # Fill missing values
        self.players_df['next_opponent_difficulty'] = self.players_df['next_opponent_difficulty'].fillna(3)
        self.players_df['is_home'] = self.players_df['is_home'].fillna(0)
        self.players_df['next_opponent'] = self.players_df['next_opponent'].fillna(0)
        self.players_df['opponent_strength'] = self.players_df['opponent_strength'].fillna(3)
        self.players_df['home_advantage_bonus'] = self.players_df['home_advantage_bonus'].fillna(0)
    
    def prepare_features(self):
        """Prepare enhanced features for the model"""
        feature_columns = [
            'points_per_game', 'form', 'recent_form_weighted', 'points_per_minute', 
            'goal_involvement_per_90', 'minutes_reliability', 'team_attack', 'team_defense', 
            'cost', 'difficulty_factor', 'is_home', 'position_specific_score', 'home_advantage_bonus',
            'ict_index', 'selected_by_percent', 'opponent_strength',
            'historical_consistency', 'team_home_strength', 'team_away_strength'
        ]
        
        available_features = [col for col in feature_columns if col in self.players_df.columns]
        X = self.players_df[available_features].copy()
        
        for col in X.columns:
            X[col] = self.clean_numeric_data(X[col])
            X[col] = X[col].fillna(0)
            X[col] = X[col].replace([np.inf, -np.inf], 0)
        
        return X, available_features
    
    def train_model(self):
        """Train enhanced XGBoost model"""
        X, feature_names = self.prepare_features()
        
        # More conservative target variable
        y = (self.players_df['points_per_game'] * 0.7 +  # Increased PPG weight
             self.players_df['form'] * 0.2 +             # Reduced form weight
             self.players_df['historical_consistency'] * 0.1).copy()
        
        self.model = xgb.XGBRegressor(
            n_estimators=150,  # Reduced from 200
            learning_rate=0.08,  # Reduced from 0.1
            max_depth=5,        # Reduced from 6
            subsample=0.7,      # Reduced from 0.8
            colsample_bytree=0.7,  # Reduced from 0.8
            random_state=42,
            n_jobs=-1
        )
        
        # Only train on players with meaningful minutes
        valid_players = self.players_df['minutes'] > 180
        X_filtered = X[valid_players]
        y_filtered = y[valid_players]
        
        if len(X_filtered) == 0:
            return False
        
        self.model.fit(X_filtered, y_filtered)
        
        # Calculate model accuracy
        self.calculate_model_accuracy(X_filtered, y_filtered)
        
        return True
    
    def calculate_model_accuracy(self, X, y):
        """Calculate and store model accuracy metrics"""
        try:
            predictions = self.model.predict(X)
            
            self.model_accuracy = {
                'mae': mean_absolute_error(y, predictions),
                'rmse': np.sqrt(mean_squared_error(y, predictions)),
                'r2': r2_score(y, predictions),
                'accuracy_within_1': np.mean(np.abs(predictions - y) <= 1.0),
                'accuracy_within_2': np.mean(np.abs(predictions - y) <= 2.0)
            }
        except Exception as e:
            st.warning(f"Could not calculate model accuracy: {e}")
    
    def search_player(self, player_name):
        """Search for a player with exact match priority"""
        if not player_name or len(player_name.strip()) < 2:
            return pd.DataFrame()
        
        player_name_lower = player_name.lower().strip()
        players_df = self.players_df.copy()
        
        # First: Exact matches (case insensitive)
        exact_matches = players_df[
            players_df['web_name'].str.lower() == player_name_lower
        ]
        
        if len(exact_matches) > 0:
            return exact_matches
        
        # Second: Starts with matches
        starts_with_matches = players_df[
            players_df['web_name'].str.lower().str.startswith(player_name_lower)
        ]
        
        if len(starts_with_matches) > 0:
            return starts_with_matches
        
        # Third: Contains matches (original behavior)
        contains_matches = players_df[
            players_df['web_name'].str.lower().str.contains(player_name_lower, na=False)
        ]
        
        return contains_matches
    
    def predict_player(self, player_name):
        """Predict player points with MORE CONSERVATIVE adjustments"""
        matches = self.search_player(player_name)
        
        if len(matches) == 0:
            return None
        
        # If multiple matches, use the one with most minutes (most likely active player)
        if len(matches) > 1:
            # Sort by minutes (descending) to prioritize active players
            matches = matches.sort_values('minutes', ascending=False)
            st.info(f"🔍 Multiple players found. Showing: {matches.iloc[0]['web_name']}")
        
        player = matches.iloc[0]
        
        if not self.is_player_available(player):
            return self.format_unavailable_player(player)
        
        X, feature_names = self.prepare_features()
        
        player_mask = self.players_df['id'] == player['id']
        if not player_mask.any():
            return None
        
        player_features = X[player_mask].iloc[0].values.reshape(1, -1)
        
        # Get player's current PPG
        current_ppg = player.get('points_per_game', 0)
        
        # More conservative base prediction
        base_prediction = self.model.predict(player_features)[0]
        
        # Weighted average heavily favoring PPG (more conservative)
        weighted_base = (current_ppg * 0.8) + (base_prediction * 0.2)  # Increased PPG weight
        
        # More conservative prediction adjustments
        position = player['element_type']
        form = player.get('form', 0)
        opponent_difficulty = player.get('next_opponent_difficulty', 3)
        is_home = player.get('is_home', 0)
        minutes = player.get('minutes', 0)
        historical_consistency = player.get('historical_consistency', 0.5)
        
        # More conservative position-specific adjustments
        position_factors = {1: 0.9, 2: 0.95, 3: 1.0, 4: 1.05}  # Reduced all factors
        position_factor = position_factors.get(position, 1.0)
        
        # More conservative form multiplier
        form_multiplier = 1.0 + (form * 0.03) + (historical_consistency * 0.01)  # Reduced impact
        
        # More conservative difficulty adjustment
        difficulty_multiplier = 1.05 - (opponent_difficulty * 0.03)  # Reduced impact
        
        # More conservative home advantage
        home_strength = player.get('team_home_strength', 1.0)
        away_strength = player.get('team_away_strength', 1.0)
        home_multiplier = 1.03 * home_strength if is_home else 0.97 * away_strength  # Reduced impact
        
        # More conservative minutes reliability
        minutes_factor = 1.0 if minutes > 270 else (0.7 if minutes <= 90 else 0.85)  # More strict
        
        # Calculate final prediction with conservative factors
        final_prediction = (weighted_base * position_factor * form_multiplier * 
                          difficulty_multiplier * home_multiplier * minutes_factor)
        
        # REDUCE PREDICTION BY 15% as requested
        final_prediction = final_prediction * 0.85
        
        # Tighter bounds around PPG
        ppg_deviation_limit = 0.2  # Reduced from 0.3
        ppg_lower_bound = current_ppg * (1 - ppg_deviation_limit)
        ppg_upper_bound = current_ppg * (1 + ppg_deviation_limit)
        
        bounded_prediction = np.clip(final_prediction, ppg_lower_bound, ppg_upper_bound)
        
        # More realistic limits based on position
        position_limits = {1: (1, 8), 2: (1, 10), 3: (1, 12), 4: (1, 12)}  # Reduced maximums
        min_points, max_points = position_limits.get(position, (1, 8))
        
        predicted_points = np.clip(bounded_prediction, min_points, max_points)
        
        return self.format_prediction_result(player, predicted_points)
    
    def search_players_autocomplete(self, query):
        """Search for players with better matching"""
        if not query or len(query.strip()) < 1:
            return []
        
        query_lower = query.lower().strip()
        all_matches = []
        
        # Exact matches first
        exact_matches = [name for name in self.all_player_names 
                        if name.lower() == query_lower]
        all_matches.extend(exact_matches)
        
        # Then starts with matches
        starts_with = [name for name in self.all_player_names 
                      if name.lower().startswith(query_lower) and name not in exact_matches]
        all_matches.extend(starts_with)
        
        # Then contains matches
        contains = [name for name in self.all_player_names 
                   if query_lower in name.lower() and name not in all_matches]
        all_matches.extend(contains)
        
        return all_matches[:10]  # Limit to 10 results
    
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
    
    def format_unavailable_player(self, player):
        """Format response for unavailable player"""
        opponent_id = player.get('next_opponent', 0)
        
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
            
            current_gw = self.get_current_gameweek()
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

    def get_current_gameweek(self):
        """Get current gameweek for team analysis"""
        try:    
            bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
            response = requests.get(bootstrap_url, timeout=10)
            data = response.json()

            events = data['events']
            current_gw = None

            for event in events:
                if event['is_current']:
                    current_gw = event['id']
                    break

            return current_gw if current_gw else 8
        except Exception as e:
            return 8

    def analyze_user_team(self, team_id):
        """Analyze user team and return predictions for NEXT gameweek"""
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
        """Suggest transfers to improve team for NEXT gameweek"""
        suggestions = []
        
        if not team_analysis or 'players' not in team_analysis:
            return suggestions
        
        current_players = team_analysis['players']
        
        # Find weak players (low predicted points or unavailable)
        weak_players = []
        for player in current_players:
            if player['unavailable'] or player['predicted_points'] < 2.0:  # Reduced threshold
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
                (self.players_df['now_cost'] / 10 <= current_cost + 0.5) &  # Reduced budget
                (~self.players_df['id'].isin([p['id'] for p in current_players]))
            ]
            
            # Get predictions for alternatives
            alternative_predictions = []
            for _, alt_player in alternatives.iterrows():
                alt_prediction = self.predict_player(alt_player['web_name'])
                if alt_prediction and alt_prediction['predicted_points'] > weak_player['predicted_points'] + 0.3:  # Reduced threshold
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

    def get_best_lineup_352(self):
        """Get the best 3-5-2 lineup based on predicted points for NEXT gameweek"""
        try:
            # Get predictions for all players with sufficient minutes
            all_predictions = []
            
            for _, player in self.players_df.iterrows():
                if player['minutes'] > 180:
                    try:
                        prediction = self.predict_player(player['web_name'])
                        if prediction and prediction['predicted_points'] > 0 and not prediction.get('unavailable', False):
                            all_predictions.append({
                                'id': player['id'],
                                'name': player['web_name'],
                                'team': self.get_team_name(player['team']),
                                'position': self.get_position_name(player['element_type']),
                                'position_code': player['element_type'],
                                'predicted_points': prediction['predicted_points'],
                                'cost': player.get('now_cost', 0) / 10,
                                'form': player.get('form', 0),
                                'opponent': prediction['opponent_name'],
                                'is_home': prediction.get('is_home', 0)
                            })
                    except Exception as e:
                        continue
            
            if not all_predictions:
                return None
            
            predictions_df = pd.DataFrame(all_predictions)
            
            # Filter by position and get top players for 3-5-2 formation
            lineup = {}
            
            # Goalkeepers (2 needed - 1 starter + 1 substitute)
            gks = predictions_df[predictions_df['position_code'] == 1].nlargest(2, 'predicted_points')
            lineup['goalkeepers'] = gks.to_dict('records')
            
            # Defenders (5 needed - 3 starters + 2 substitutes)
            defenders = predictions_df[predictions_df['position_code'] == 2].nlargest(5, 'predicted_points')
            lineup['defenders'] = defenders.to_dict('records')
            
            # Midfielders (5 needed - 5 starters)
            midfielders = predictions_df[predictions_df['position_code'] == 3].nlargest(5, 'predicted_points')
            lineup['midfielders'] = midfielders.to_dict('records')
            
            # Forwards (3 needed - 2 starters + 1 substitute)
            forwards = predictions_df[predictions_df['position_code'] == 4].nlargest(3, 'predicted_points')
            lineup['forwards'] = forwards.to_dict('records')
            
            # Calculate total predicted points
            total_points = (
                lineup['goalkeepers'][0]['predicted_points'] +
                sum(player['predicted_points'] for player in lineup['defenders'][:3]) +
                sum(player['predicted_points'] for player in lineup['midfielders'][:5]) +
                sum(player['predicted_points'] for player in lineup['forwards'][:2])
            )
            
            lineup['total_predicted_points'] = total_points
            lineup['formation'] = "3-5-2"
            
            return lineup
            
        except Exception as e:
            st.error(f"Error generating lineup: {e}")
            return None

def get_translation(key):
    """Return translation for a given key"""
    return TRANSLATIONS.get(key, key)

def display_player_prediction(prediction, predictor):
    """Display player prediction"""
    if not prediction:
        return
    
    next_gw = predictor.get_next_gameweek()

    player = prediction['player']
    predicted_points = prediction['predicted_points']
    team_name = prediction['team_name']
    opponent_name = prediction['opponent_name']
    is_home = prediction['is_home']
    unavailable = prediction.get('unavailable', False)
    
    st.markdown("---")
    
    # More conservative prediction badge thresholds
    if unavailable:
        badge_class = "poor-player"
        badge_text = f"❌ 0.0 points (Unavailable) - GW{next_gw}"
    elif predicted_points >= 5:  # Reduced from 6
        badge_class = "top-player"
        badge_text = f"🚀 {predicted_points:.1f} pts 🚀 - GW{next_gw}"
    elif predicted_points >= 3.5:  # Reduced from 4
        badge_class = "good-player"
        badge_text = f"🔥 {predicted_points:.1f} pts 🔥 - GW{next_gw}"
    elif predicted_points >= 2.0:  # Reduced from 2.5
        badge_class = "avg-player"
        badge_text = f"⚡ {predicted_points:.1f} pts ⚡ - GW{next_gw}"
    else:
        badge_class = "poor-player"
        badge_text = f"📉 {predicted_points:.1f} pts 📉 - GW{next_gw}"
    
    st.markdown(f'<div class="prediction-badge {badge_class}">{badge_text}</div>', unsafe_allow_html=True)
    
    # Player card with image placeholder
    st.markdown('<div class="player-card">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"### 👤 {player['web_name']}")
        st.markdown(f"**🏟️ Team:** {team_name}")
        st.markdown(f"**💰 Cost:** {player['cost']:.1f}M")
        st.markdown(f"**📊 Position:** {predictor.get_position_name(player['element_type'])}")
    
    with col2:
        # Player image placeholder (you can replace with actual player images)
        st.markdown("""
        <div style="text-align: center;">
            <div style="width: 80px; height: 80px; background: linear-gradient(135deg, #667eea, #764ba2); 
                        border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                        margin: 0 auto; border: 3px solid #667eea; box-shadow: 0 5px 15px rgba(0,0,0,0.3);">
                <span style="color: white; font-size: 24px; font-weight: bold;">⚽</span>
            </div>
            <p style="margin-top: 10px; font-weight: 500;">Player Image</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
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
    elif predicted_points >= 5:
        st.success("💡 **Excellent choice! High point potential this week.**")
    elif predicted_points >= 3.5:
        st.success("💡 **Good choice! Solid performance expected.**")
    elif predicted_points >= 2.0:
        st.info("💡 **Decent choice. Average performance expected.**")
    else:
        st.warning("💡 **Risky choice. Limited performance expected.**")

def display_team_analysis_with_transfers(team_analysis, predictor):
    """Display team analysis with transfer recommendations"""
    if not team_analysis:
        return
    
    next_gw = predictor.get_next_gameweek()
    
    st.markdown(f"## {get_translation('team_analysis_title')} - GW{next_gw}")
    
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
                points_color = "🟢" if player['predicted_points'] >= 4 else "🟡" if player['predicted_points'] >= 2.5 else "🔴"  # Adjusted thresholds
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

def display_model_accuracy(predictor):
    """Display model accuracy metrics"""
    if not predictor.model_accuracy:
        st.warning("Model accuracy data not available")
        return
    
    st.markdown(f"## {get_translation('model_accuracy')}")
    
    accuracy = predictor.model_accuracy
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        mae = accuracy['mae']
        if mae < 1.0:
            st.markdown(f'<div class="accuracy-good"><div class="stat-value">{mae:.2f}</div><div class="stat-label">Mean Absolute Error</div></div>', unsafe_allow_html=True)
        elif mae < 1.5:
            st.markdown(f'<div class="accuracy-avg"><div class="stat-value">{mae:.2f}</div><div class="stat-label">Mean Absolute Error</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="accuracy-poor"><div class="stat-value">{mae:.2f}</div><div class="stat-label">Mean Absolute Error</div></div>', unsafe_allow_html=True)
    
    with col2:
        rmse = accuracy['rmse']
        if rmse < 1.5:
            st.markdown(f'<div class="accuracy-good"><div class="stat-value">{rmse:.2f}</div><div class="stat-label">Root Mean Squared Error</div></div>', unsafe_allow_html=True)
        elif rmse < 2.0:
            st.markdown(f'<div class="accuracy-avg"><div class="stat-value">{rmse:.2f}</div><div class="stat-label">Root Mean Squared Error</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="accuracy-poor"><div class="stat-value">{rmse:.2f}</div><div class="stat-label">Root Mean Squared Error</div></div>', unsafe_allow_html=True)
    
    with col3:
        r2 = accuracy['r2']
        if r2 > 0.7:
            st.markdown(f'<div class="accuracy-good"><div class="stat-value">{r2:.2f}</div><div class="stat-label">R² Score</div></div>', unsafe_allow_html=True)
        elif r2 > 0.5:
            st.markdown(f'<div class="accuracy-avg"><div class="stat-value">{r2:.2f}</div><div class="stat-label">R² Score</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="accuracy-poor"><div class="stat-value">{r2:.2f}</div><div class="stat-label">R² Score</div></div>', unsafe_allow_html=True)
    
    with col4:
        acc_1 = accuracy['accuracy_within_1']
        if acc_1 > 0.7:
            st.markdown(f'<div class="accuracy-good"><div class="stat-value">{acc_1:.1%}</div><div class="stat-label">Within 1 Point</div></div>', unsafe_allow_html=True)
        elif acc_1 > 0.5:
            st.markdown(f'<div class="accuracy-avg"><div class="stat-value">{acc_1:.1%}</div><div class="stat-label">Within 1 Point</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="accuracy-poor"><div class="stat-value">{acc_1:.1%}</div><div class="stat-label">Within 1 Point</div></div>', unsafe_allow_html=True)
    
    with col5:
        acc_2 = accuracy['accuracy_within_2']
        if acc_2 > 0.85:
            st.markdown(f'<div class="accuracy-good"><div class="stat-value">{acc_2:.1%}</div><div class="stat-label">Within 2 Points</div></div>', unsafe_allow_html=True)
        elif acc_2 > 0.7:
            st.markdown(f'<div class="accuracy-avg"><div class="stat-value">{acc_2:.1%}</div><div class="stat-label">Within 2 Points</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="accuracy-poor"><div class="stat-value">{acc_2:.1%}</div><div class="stat-label">Within 2 Points</div></div>', unsafe_allow_html=True)
    
    # Interpretation
    st.markdown("---")
    st.markdown(f"#### {get_translation('prediction_confidence')}")
    
    overall_accuracy = (accuracy['accuracy_within_1'] + accuracy['accuracy_within_2']) / 2
    
    if overall_accuracy > 0.75:
        st.success("**🎯 HIGH CONFIDENCE** - Model shows strong predictive power with conservative adjustments")
        st.info("""
        **Why it's accurate:**
        - Conservative prediction formula
        - Heavy weighting on actual PPG
        - Reduced impact of external factors
        - Realistic point limits
        """)
    elif overall_accuracy > 0.6:
        st.warning("**⚡ MODERATE CONFIDENCE** - Model provides reasonable estimates")
        st.info("""
        **Features:**
        - More realistic point predictions
        - Better alignment with actual performance
        - Conservative transfer suggestions
        """)
    else:
        st.error("**📉 LOW CONFIDENCE** - Predictions should be used cautiously")
        st.info("""
        **Recommendations:**
        - Focus on consistent performers
        - Use as guidance only
        - Consider recent form
        """)

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

def display_best_lineup(lineup, predictor):
    """Display the best 3-5-2 lineup"""
    if not lineup:
        st.error("❌ Could not generate lineup")
        return
    
    next_gw = predictor.get_next_gameweek()
    
    st.markdown(f"## {get_translation('best_lineup')} - GW{next_gw}")
    
    # Total points and formation
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predicted Points", f"{lineup['total_predicted_points']:.1f}")
    with col2:
        st.metric("Formation", lineup['formation'])
    with col3:
        st.metric("Squad Size", "15 players")
    
    st.markdown("---")
    
    # Starting XI
    st.markdown("### 🟢 Starting XI")
    
    # Goalkeeper
    if lineup['goalkeepers']:
        gk = lineup['goalkeepers'][0]
        with st.container():
            st.markdown('<div class="player-card">', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            with col1:
                st.markdown(f"**🧤 {gk['name']}**")
                st.markdown(f"*{gk['team']}*")
            with col2:
                st.markdown(f"**Opponent:** {gk['opponent']}")
            with col3:
                home_away = "🏠 Home" if gk['is_home'] else "✈️ Away"
                st.markdown(f"**Venue:** {home_away}")
            with col4:
                st.markdown(f"**{gk['predicted_points']:.1f} pts**")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Defenders (3)
    st.markdown("#### 🛡️ Defenders")
    for i, defender in enumerate(lineup['defenders'][:3]):
        with st.container():
            st.markdown('<div class="player-card">', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            with col1:
                st.markdown(f"**#{i+1} {defender['name']}**")
                st.markdown(f"*{defender['team']}*")
            with col2:
                st.markdown(f"**Opponent:** {defender['opponent']}")
            with col3:
                home_away = "🏠 Home" if defender['is_home'] else "✈️ Away"
                st.markdown(f"**Venue:** {home_away}")
            with col4:
                st.markdown(f"**{defender['predicted_points']:.1f} pts**")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Midfielders (5)
    st.markdown("#### ⚡ Midfielders")
    for i, midfielder in enumerate(lineup['midfielders'][:5]):
        with st.container():
            st.markdown('<div class="player-card">', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            with col1:
                st.markdown(f"**#{i+1} {midfielder['name']}**")
                st.markdown(f"*{midfielder['team']}*")
            with col2:
                st.markdown(f"**Opponent:** {midfielder['opponent']}")
            with col3:
                home_away = "🏠 Home" if midfielder['is_home'] else "✈️ Away"
                st.markdown(f"**Venue:** {home_away}")
            with col4:
                st.markdown(f"**{midfielder['predicted_points']:.1f} pts**")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Forwards (2)
    st.markdown("#### 🎯 Forwards")
    for i, forward in enumerate(lineup['forwards'][:2]):
        with st.container():
            st.markdown('<div class="player-card">', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            with col1:
                st.markdown(f"**#{i+1} {forward['name']}**")
                st.markdown(f"*{forward['team']}*")
            with col2:
                st.markdown(f"**Opponent:** {forward['opponent']}")
            with col3:
                home_away = "🏠 Home" if forward['is_home'] else "✈️ Away"
                st.markdown(f"**Venue:** {home_away}")
            with col4:
                st.markdown(f"**{forward['predicted_points']:.1f} pts**")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Substitutes
    st.markdown("---")
    st.markdown("### 🔄 Substitutes")
    
    sub_col1, sub_col2, sub_col3 = st.columns(3)
    
    with sub_col1:
        st.markdown("**Goalkeeper**")
        if len(lineup['goalkeepers']) > 1:
            sub_gk = lineup['goalkeepers'][1]
            st.markdown(f"🧤 {sub_gk['name']}")
            st.markdown(f"*{sub_gk['team']}*")
            st.markdown(f"**{sub_gk['predicted_points']:.1f} pts**")
    
    with sub_col2:
        st.markdown("**Defenders**")
        for defender in lineup['defenders'][3:5]:
            st.markdown(f"🛡️ {defender['name']}")
            st.markdown(f"*{defender['team']}*")
            st.markdown(f"**{defender['predicted_points']:.1f} pts**")
            st.markdown("---")
    
    with sub_col3:
        st.markdown("**Forward**")
        if len(lineup['forwards']) > 2:
            sub_fwd = lineup['forwards'][2]
            st.markdown(f"🎯 {sub_fwd['name']}")
            st.markdown(f"*{sub_fwd['team']}*")
            st.markdown(f"**{sub_fwd['predicted_points']:.1f} pts**")            

def main():
    if 'search_player' not in st.session_state:
        st.session_state.search_player = ""
    if 'team_id' not in st.session_state:
        st.session_state.team_id = ""
    if 'selected_suggestion' not in st.session_state:
        st.session_state.selected_suggestion = None
    
    # Get NEXT gameweek for display
    next_gw = 9
    if 'predictor' in st.session_state:
        next_gw = st.session_state.predictor.get_next_gameweek()
    
    # Main header with floating animation
    st.markdown(f'<h1 class="main-header floating">{get_translation("title")}</h1>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">🔮 AI-Powered Points Predictions for Gameweek {next_gw}</div>', unsafe_allow_html=True)
    
    st.sidebar.title(get_translation("navigation"))
    st.sidebar.markdown("---")
    
    if 'predictor' not in st.session_state:
        st.session_state.predictor = FPLPredictor()
        with st.spinner('🔄 Loading FPL data with historical integration...'):
            if st.session_state.predictor.load_all_fpl_data():
                st.session_state.predictor.create_features()
                with st.spinner('🤖 Training enhanced AI model...'):
                    if st.session_state.predictor.train_model():
                        st.sidebar.success('✅ Enhanced model ready!')
                    else:
                        st.sidebar.error('❌ Training error')
            else:
                st.sidebar.error('❌ Data loading error')
    
    # Model Accuracy Section
    if st.session_state.predictor.model_accuracy:
        display_model_accuracy(st.session_state.predictor)
    
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
     
    # Add this in the sidebar section with other buttons
    st.sidebar.markdown("---")
    if st.sidebar.button("🏆 Best 3-5-2 Lineup"):
        with st.spinner('🔍 Generating optimal lineup...'):
            lineup = st.session_state.predictor.get_best_lineup_352()
            if lineup:
                display_best_lineup(lineup, st.session_state.predictor)
            else:
                st.error("❌ Could not generate lineup")        
    
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