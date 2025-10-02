import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="FPL Predictor Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dictionnaires de traduction
TRANSLATIONS = {
    'fr': {
        'title': "⚽ FPL Predictor Pro",
        'subtitle': "🔮 Prédictions de points pour la Gameweek 7",
        'navigation': "🎮 Navigation",
        'player_search': "🔍 Recherche de Joueur",
        'player_placeholder': "ex: Salah, Haaland...",
        'filters': "🎯 Filtres Avancés",
        'filter_team': "Filtrer par équipe:",
        'filter_position': "Filtrer par position:",
        'all_teams': "Toutes",
        'all_positions': "Toutes",
        'apply_filters': "🔍 Appliquer les Filtres",
        'view_filtered': "👥 Voir Joueurs Filtrés",
        'view_top10': "🏆 Voir le Top 10",
        'searching': "🔍 Recherche de",
        'player_not_found': "❌ Joueur non trouvé. Essaye avec un autre nom.",
        'player_card': "👤 Joueur",
        'team': "🏟️ Équipe",
        'cost': "💰 Coût",
        'position': "📊 Position",
        'stats': "📈 Statistiques Actuelles",
        'form': "Forme récente",
        'total_points': "Points totaux",
        'ppg': "Points par match",
        'goals': "Buts",
        'assists': "Passes",
        'minutes': "Minutes",
        'match_context': "📅 Contexte du Match",
        'location': "Lieu",
        'difficulty': "Difficulté",
        'opponent': "Adversaire",
        'home': "🏠 Domicile",
        'away': "✈️ Extérieur",
        'prediction': "🔮 Prédiction pour la GW7",
        'interpretation': "💡 Interprétation",
        'excellent_choice': "Excellent choix! Fort potentiel de points cette semaine.",
        'good_choice': "Bon choix! Performance solide attendue.",
        'decent_choice': "Choix décent. Performance moyenne attendue.",
        'risky_choice': "Choix risqué. Performance limitée attendue.",
        'filtered_players': "👥 Joueurs Filtrés",
        'no_players': "❌ Aucun joueur ne correspond aux critères sélectionnés.",
        'players_found': "✅ {} joueurs trouvés",
        'top_predictions': "🏆 Top 10 des Prédictions",
        'how_to_use': "💡 Comment utiliser:",
        'usage_steps': [
            "🔍 Recherche un joueur spécifique",
            "🎯 Filtres pour voir tous les joueurs d'une équipe/position",
            "🏆 Top 10 pour les meilleures prédictions",
            "👥 Joueurs Filtrés pour explorer par critères"
        ],
        'goalkeeper': "Gardien",
        'defender': "Défenseur",
        'midfielder': "Milieu",
        'forward': "Attaquant"
    },
    'en': {
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
        'forward': "Forward"
    }
}

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .player-card {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .prediction-high {
        color: #00ff00;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .prediction-medium {
        color: #ffa500;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .prediction-low {
        color: #ff0000;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .language-switcher {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 1000;
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
        
    def load_all_fpl_data(self):
        """Charge les données FPL"""
        try:
            bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
            response = requests.get(bootstrap_url)
            data = response.json()
            
            self.players_df = pd.DataFrame(data['elements'])
            self.teams_df = pd.DataFrame(data['teams'])
            
            fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"
            fixtures_response = requests.get(fixtures_url)
            self.fixtures_df = pd.DataFrame(fixtures_response.json())
            
            return True
        except:
            return False
    
    def clean_numeric_data(self, series):
        """Nettoie les données numériques"""
        if pd.api.types.is_numeric_dtype(series):
            return series
        series_clean = series.replace([None, 'None', 'null', ''], '0')
        try:
            return pd.to_numeric(series_clean, errors='coerce')
        except:
            return pd.Series([0] * len(series), index=series.index)
    
    def create_features(self):
        """Crée les features pour le modèle"""
        # Nettoyage
        critical_columns = ['form', 'points_per_game', 'goals_scored', 'assists', 'minutes', 
                           'total_points', 'influence', 'creativity', 'threat', 'ict_index', 'bps']
        
        for col in critical_columns:
            if col in self.players_df.columns:
                self.players_df[col] = self.clean_numeric_data(self.players_df[col])
        
        # Features de base
        self.players_df['cost'] = self.players_df['now_cost'] / 10
        self.players_df['points_per_minute'] = self.players_df['total_points'] / self.players_df['minutes']
        self.players_df['points_per_minute'] = self.players_df['points_per_minute'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Impact offensif
        self.players_df['goal_involvement_per_90'] = (self.players_df['goals_scored'] + self.players_df['assists']) / (self.players_df['minutes'] / 90)
        self.players_df['goal_involvement_per_90'] = self.players_df['goal_involvement_per_90'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Consistance
        self.players_df['consistency_score'] = self.players_df['points_per_game'] / self.players_df['points_per_game'].max()
        
        # Minutes stability
        self.players_df['minutes_ratio'] = self.players_df['minutes'] / (6 * 90)
        self.players_df['minutes_ratio'] = self.players_df['minutes_ratio'].clip(0, 1)
        
        # Features d'équipe
        team_strength = dict(zip(self.teams_df['id'], self.teams_df['strength']))
        team_attack = dict(zip(self.teams_df['id'], self.teams_df['strength_attack_home']))
        self.players_df['team_strength'] = self.players_df['team'].map(team_strength)
        self.players_df['team_attack'] = self.players_df['team'].map(team_attack)
        
        # Contexte du match
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
            
            # Mapping adversaire
            opponent_mapping[team_h] = team_a
            opponent_mapping[team_a] = team_h
        
        self.players_df['next_opponent_difficulty'] = self.players_df['team'].map(team_difficulty)
        self.players_df['is_home'] = self.players_df['team'].map(home_advantage)
        self.players_df['next_opponent'] = self.players_df['team'].map(opponent_mapping)
        self.players_df['difficulty_factor'] = (6 - self.players_df['next_opponent_difficulty']) / 5
    
    def prepare_features(self):
        """Prépare les features pour le modèle"""
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
        """Entraîne le modèle"""
        X, feature_names = self.prepare_features()
        y = self.players_df['points_per_game'].copy()
        
        # Ajustements
        minutes_threshold = 180
        low_minutes_players = self.players_df['minutes'] < minutes_threshold
        y[low_minutes_players] = y[low_minutes_players] * 0.7
        y = y.clip(upper=12.0)
        
        # Filtrer les joueurs valides
        valid_players = self.players_df['minutes'] > 90
        X_filtered = X[valid_players]
        y_filtered = y[valid_players]
        
        if len(X_filtered) == 0:
            return False
        
        # Entraînement
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
        """Recherche un joueur"""
        player_name_lower = player_name.lower()
        matches = self.players_df[
            self.players_df['web_name'].str.lower().str.contains(player_name_lower, na=False)
        ]
        return matches
    
    def predict_player(self, player_name):
        """Prédit les points d'un joueur"""
        matches = self.search_player(player_name)
        
        if len(matches) == 0:
            return None
        
        player = matches.iloc[0]
        X, feature_names = self.prepare_features()
        
        player_mask = self.players_df['id'] == player['id']
        if not player_mask.any():
            return None
        
        player_features = X[player_mask].iloc[0].values.reshape(1, -1)
        predicted_points = self.model.predict(player_features)[0]
        
        # Ajustements réalistes
        minutes_factor = player.get('minutes_ratio', 1)
        difficulty_factor = player.get('difficulty_factor', 1)
        final_prediction = predicted_points * minutes_factor * difficulty_factor
        final_prediction = np.clip(final_prediction, 0, 12)
        
        # Obtenir le nom de l'adversaire
        opponent_id = player.get('next_opponent')
        opponent_name = self.get_team_name(opponent_id) if opponent_id else "Inconnu"
        
        return {
            'player': player,
            'predicted_points': final_prediction,
            'team_name': self.get_team_name(player['team']),
            'opponent_name': opponent_name,
            'is_home': player.get('is_home', 0)
        }
    
    def get_team_name(self, team_id):
        """Retourne le nom de l'équipe"""
        team_mapping = dict(zip(self.teams_df['id'], self.teams_df['name']))
        return team_mapping.get(team_id, 'Unknown')
    
    def get_position_name(self, position_code, language='fr'):
        """Retourne le nom de la position"""
        positions_fr = {1: "Gardien", 2: "Défenseur", 3: "Milieu", 4: "Attaquant"}
        positions_en = {1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Forward"}
        
        if language == 'en':
            return positions_en.get(position_code, "Unknown")
        return positions_fr.get(position_code, "Inconnu")
    
    def get_filtered_players(self, team_filter, position_filter):
        """Filtre les joueurs selon les critères"""
        filtered_players = self.players_df.copy()
        
        # Filtre par équipe
        if team_filter != "Toutes" and team_filter != "All":
            team_id = self.teams_df[self.teams_df['name'] == team_filter]['id'].values
            if len(team_id) > 0:
                filtered_players = filtered_players[filtered_players['team'] == team_id[0]]
        
        # Filtre par position
        if position_filter != "Toutes" and position_filter != "All":
            position_map_fr = {"Gardien": 1, "Défenseur": 2, "Milieu": 3, "Attaquant": 4}
            position_map_en = {"Goalkeeper": 1, "Defender": 2, "Midfielder": 3, "Forward": 4}
            
            position_code = position_map_fr.get(position_filter) or position_map_en.get(position_filter)
            if position_code:
                filtered_players = filtered_players[filtered_players['element_type'] == position_code]
        
        return filtered_players

def get_translation(key, language='fr'):
    """Retourne la traduction pour une clé donnée"""
    return TRANSLATIONS[language].get(key, key)

# Interface Streamlit
def main():
    # Sélecteur de langue
    col_lang1, col_lang2, col_lang3 = st.columns([1, 2, 1])
    with col_lang2:
        language = st.radio(
            "🌍 Language / Langue:",
            ['fr', 'en'],
            index=0,
            horizontal=True,
            label_visibility="collapsed"
        )
    
    # Header avec traduction
    st.markdown(f'<h1 class="main-header">{get_translation("title", language)}</h1>', unsafe_allow_html=True)
    st.markdown(f"### {get_translation('subtitle', language)}")
    
    # Sidebar
    st.sidebar.title(get_translation("navigation", language))
    st.sidebar.markdown("---")
    
    # Initialisation du modèle
    if 'predictor' not in st.session_state:
        st.session_state.predictor = FPLWebPredictor()
        with st.spinner('🔄 Chargement des données FPL...' if language == 'fr' else '🔄 Loading FPL data...'):
            if st.session_state.predictor.load_all_fpl_data():
                st.session_state.predictor.create_features()
                with st.spinner('🤖 Entraînement du modèle...' if language == 'fr' else '🤖 Training model...'):
                    if st.session_state.predictor.train_model():
                        st.sidebar.success('✅ Modèle prêt!' if language == 'fr' else '✅ Model ready!')
                    else:
                        st.sidebar.error('❌ Erreur entraînement' if language == 'fr' else '❌ Training error')
            else:
                st.sidebar.error('❌ Erreur chargement données' if language == 'fr' else '❌ Data loading error')
    
    # Recherche de joueur
    st.sidebar.markdown(f"## {get_translation('player_search', language)}")
    player_name = st.sidebar.text_input(
        get_translation('player_search', language) + ":",
        placeholder=get_translation('player_placeholder', language)
    )
    
    # Filtres AVEC FONCTIONNALITÉ
    st.sidebar.markdown(f"## {get_translation('filters', language)}")
    
    # Filtre par équipe
    team_options = [get_translation('all_teams', language)] + sorted(st.session_state.predictor.teams_df['name'].tolist())
    team_filter = st.sidebar.selectbox(
        get_translation('filter_team', language),
        team_options
    )
    
    # Filtre par position
    position_options_fr = [get_translation('all_positions', language), "Gardien", "Défenseur", "Milieu", "Attaquant"]
    position_options_en = [get_translation('all_positions', language), "Goalkeeper", "Defender", "Midfielder", "Forward"]
    position_options = position_options_fr if language == 'fr' else position_options_en
    
    position_filter = st.sidebar.selectbox(
        get_translation('filter_position', language),
        position_options
    )
    
    # Bouton pour appliquer les filtres
    apply_filters = st.sidebar.button(get_translation('apply_filters', language))
    
    # Section recherche principale
    if player_name:
        with st.spinner(f'{get_translation("searching", language)} {player_name}...'):
            prediction = st.session_state.predictor.predict_player(player_name)
            
            if prediction:
                player = prediction['player']
                predicted_points = prediction['predicted_points']
                team_name = prediction['team_name']
                opponent_name = prediction['opponent_name']
                is_home = prediction['is_home']
                
                # Carte du joueur
                st.markdown(f'<div class="player-card">', unsafe_allow_html=True)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"### {get_translation('player_card', language)}: {player['web_name']}")
                    st.markdown(f"**{get_translation('team', language)}:** {team_name}")
                    st.markdown(f"**{get_translation('cost', language)}:** {player['cost']:.1f}M")
                    st.markdown(f"**{get_translation('position', language)}:** {st.session_state.predictor.get_position_name(player['element_type'], language)}")
                
                with col2:
                    # Affichage des points prédits
                    if predicted_points >= 8:
                        points_class = "prediction-high"
                        emoji = "🔥🔥🔥"
                    elif predicted_points >= 6:
                        points_class = "prediction-medium"
                        emoji = "🔥🔥"
                    else:
                        points_class = "prediction-low"
                        emoji = "🔥"
                    
                    st.markdown(f'<div class="{points_class}">{emoji} {predicted_points:.1f} points {emoji}</div>', unsafe_allow_html=True)
                
                # Statistiques détaillées
                st.markdown("---")
                st.markdown(f"#### {get_translation('stats', language)}")
                col3, col4, col5 = st.columns(3)
                
                with col3:
                    st.metric(get_translation('form', language), f"{player['form']:.1f}")
                    st.metric(get_translation('total_points', language), f"{player['total_points']}")
                
                with col4:
                    st.metric(get_translation('ppg', language), f"{player['points_per_game']:.1f}")
                    st.metric(get_translation('goals', language), f"{player['goals_scored']}")
                
                with col5:
                    st.metric(get_translation('assists', language), f"{player['assists']}")
                    st.metric(get_translation('minutes', language), f"{player['minutes']}")
                
                # Contexte du match AMÉLIORÉ
                st.markdown("---")
                st.markdown(f"#### {get_translation('match_context', language)}")
                
                difficulty = player.get('next_opponent_difficulty', 3)
                
                # Textes de difficulté selon la langue
                if language == 'fr':
                    difficulty_text = {1: "Très Facile 🟢", 2: "Facile 🟡", 3: "Moyen 🟠", 4: "Difficile 🔴", 5: "Très Difficile 🛑"}
                    location_text = get_translation('home', language) if is_home == 1 else get_translation('away', language)
                    vs_text = "contre"
                else:
                    difficulty_text = {1: "Very Easy 🟢", 2: "Easy 🟡", 3: "Medium 🟠", 4: "Difficult 🔴", 5: "Very Difficult 🛑"}
                    location_text = get_translation('home', language) if is_home == 1 else get_translation('away', language)
                    vs_text = "vs"
                
                col6, col7, col8 = st.columns(3)
                with col6:
                    st.info(f"**{get_translation('location', language)}:** {location_text}")
                with col7:
                    st.info(f"**{get_translation('difficulty', language)}:** {difficulty_text.get(difficulty, 'Moyen' if language == 'fr' else 'Medium')}")
                with col8:
                    st.info(f"**{get_translation('opponent', language)}:** {vs_text} {opponent_name}")
                
                # Prédiction et interprétation
                st.markdown("---")
                st.markdown(f"#### {get_translation('prediction', language)}")
                
                # Interprétation selon la langue
                if language == 'fr':
                    if predicted_points >= 8:
                        interpretation = get_translation('excellent_choice', language)
                    elif predicted_points >= 6:
                        interpretation = get_translation('good_choice', language)
                    elif predicted_points >= 4:
                        interpretation = get_translation('decent_choice', language)
                    else:
                        interpretation = get_translation('risky_choice', language)
                else:
                    if predicted_points >= 8:
                        interpretation = get_translation('excellent_choice', language)
                    elif predicted_points >= 6:
                        interpretation = get_translation('good_choice', language)
                    elif predicted_points >= 4:
                        interpretation = get_translation('decent_choice', language)
                    else:
                        interpretation = get_translation('risky_choice', language)
                
                st.markdown(f"**{get_translation('interpretation', language)}:** {interpretation}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error(get_translation('player_not_found', language))
    
    # Section joueurs filtrés
    st.sidebar.markdown("---")
    if apply_filters or st.sidebar.button(get_translation('view_filtered', language)):
        st.markdown(f"## {get_translation('filtered_players', language)}")
        
        # Appliquer les filtres
        filtered_players = st.session_state.predictor.get_filtered_players(team_filter, position_filter)
        
        if len(filtered_players) == 0:
            st.warning(get_translation('no_players', language))
        else:
            st.success(get_translation('players_found', language).format(len(filtered_players)))
            
            # Afficher un échantillon des joueurs filtrés
            display_players = filtered_players[['web_name', 'team', 'element_type', 'form', 'points_per_game', 'total_points']].head(20)
            
            # Ajouter les noms d'équipes et positions
            display_players['Team'] = display_players['team'].map(
                dict(zip(st.session_state.predictor.teams_df['id'], st.session_state.predictor.teams_df['name']))
            )
            display_players['Position'] = display_players['element_type'].map(
                lambda x: st.session_state.predictor.get_position_name(x, language)
            )
            
            # Colonnes selon la langue
            if language == 'fr':
                display_columns = {
                    'web_name': 'Joueur',
                    'Team': 'Équipe', 
                    'Position': 'Position',
                    'form': 'Forme',
                    'points_per_game': 'PPG',
                    'total_points': 'Pts Totaux'
                }
            else:
                display_columns = {
                    'web_name': 'Player',
                    'Team': 'Team',
                    'Position': 'Position',
                    'form': 'Form',
                    'points_per_game': 'PPG',
                    'total_points': 'Total Points'
                }
            
            # Afficher le tableau
            st.dataframe(
                display_players.rename(columns=display_columns)
                .style.format({'Forme': '{:.1f}', 'PPG': '{:.1f}', 'Form': '{:.1f}'})
                .background_gradient(subset=['Forme', 'PPG', 'Form'], cmap='YlOrRd'),
                use_container_width=True
            )
    
    # Top 10 prédictions
    st.sidebar.markdown("---")
    if st.sidebar.button(get_translation('view_top10', language)):
        st.markdown(f"## {get_translation('top_predictions', language)}")
        
        # Calculer les prédictions pour tous les joueurs
        all_predictions = []
        X, feature_names = st.session_state.predictor.prepare_features()
        
        for _, player in st.session_state.predictor.players_df.iterrows():
            if player['minutes'] > 90:  # Seulement les joueurs actifs
                player_mask = st.session_state.predictor.players_df['id'] == player['id']
                player_features = X[player_mask].iloc[0].values.reshape(1, -1)
                predicted_points = st.session_state.predictor.model.predict(player_features)[0]
                
                # Ajustements
                minutes_factor = player.get('minutes_ratio', 1)
                difficulty_factor = player.get('difficulty_factor', 1)
                final_prediction = predicted_points * minutes_factor * difficulty_factor
                final_prediction = np.clip(final_prediction, 0, 12)
                
                all_predictions.append({
                    'Player' if language == 'en' else 'Joueur': player['web_name'],
                    'Team' if language == 'en' else 'Équipe': st.session_state.predictor.get_team_name(player['team']),
                    'Position': st.session_state.predictor.get_position_name(player['element_type'], language),
                    'Predicted Points' if language == 'en' else 'Points Prédits': final_prediction,
                    'Form' if language == 'en' else 'Forme': player['form'],
                    'Cost' if language == 'en' else 'Coût': player['cost']
                })
        
        # Créer le dataframe et trier
        top_df = pd.DataFrame(all_predictions)
        points_column = 'Predicted Points' if language == 'en' else 'Points Prédits'
        top_df = top_df.nlargest(10, points_column)
        
        # Afficher le tableau
        st.dataframe(top_df.style.format({
            'Predicted Points': '{:.1f}',
            'Points Prédits': '{:.1f}',
            'Form': '{:.1f}',
            'Forme': '{:.1f}',
            'Cost': '{:.1f}',
            'Coût': '{:.1f}'
        }).background_gradient(subset=[points_column], cmap='YlOrRd'), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(f"### {get_translation('how_to_use', language)}")
    
    for step in get_translation('usage_steps', language):
        st.markdown(f"- {step}")

if __name__ == "__main__":
    main()