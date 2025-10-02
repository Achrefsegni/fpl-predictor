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
    .team-logo {
        font-size: 2rem;
        margin-right: 10px;
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
        
        for _, fixture in next_gw_fixtures.iterrows():
            team_difficulty[fixture['team_h']] = fixture['team_h_difficulty']
            team_difficulty[fixture['team_a']] = fixture['team_a_difficulty']
            home_advantage[fixture['team_h']] = 1
            home_advantage[fixture['team_a']] = 0
        
        self.players_df['next_opponent_difficulty'] = self.players_df['team'].map(team_difficulty)
        self.players_df['is_home'] = self.players_df['team'].map(home_advantage)
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
        
        return {
            'player': player,
            'predicted_points': final_prediction,
            'team_name': self.get_team_name(player['team'])
        }
    
    def get_team_name(self, team_id):
        """Retourne le nom de l'équipe"""
        team_mapping = dict(zip(self.teams_df['id'], self.teams_df['name']))
        return team_mapping.get(team_id, 'Inconnu')
    
    def get_position_name(self, position_code):
        """Retourne le nom de la position"""
        positions = {1: "Gardien", 2: "Défenseur", 3: "Milieu", 4: "Attaquant"}
        return positions.get(position_code, "Inconnu")

# Interface Streamlit
def main():
    # Header
    st.markdown('<h1 class="main-header">⚽ FPL Predictor Pro</h1>', unsafe_allow_html=True)
    st.markdown("### 🔮 Prédictions de points pour la Gameweek 7")
    
    # Sidebar
    st.sidebar.title("🎮 Navigation")
    st.sidebar.markdown("---")
    
    # Initialisation du modèle
    if 'predictor' not in st.session_state:
        st.session_state.predictor = FPLWebPredictor()
        with st.spinner('🔄 Chargement des données FPL...'):
            if st.session_state.predictor.load_all_fpl_data():
                st.session_state.predictor.create_features()
                with st.spinner('🤖 Entraînement du modèle...'):
                    if st.session_state.predictor.train_model():
                        st.sidebar.success('✅ Modèle prêt!')
                    else:
                        st.sidebar.error('❌ Erreur entraînement')
            else:
                st.sidebar.error('❌ Erreur chargement données')
    
    # Recherche de joueur
    st.sidebar.markdown("## 🔍 Recherche")
    player_name = st.sidebar.text_input("Nom du joueur:", placeholder="ex: Salah, Haaland...")
    
    # Recherche avancée
    st.sidebar.markdown("## 🎯 Filtres")
    team_filter = st.sidebar.selectbox("Équipe:", ["Toutes"] + list(st.session_state.predictor.teams_df['name']))
    position_filter = st.sidebar.selectbox("Position:", ["Toutes", "Gardien", "Défenseur", "Milieu", "Attaquant"])
    
    # Recherche
    if player_name:
        with st.spinner(f'🔍 Recherche de {player_name}...'):
            prediction = st.session_state.predictor.predict_player(player_name)
            
            if prediction:
                player = prediction['player']
                predicted_points = prediction['predicted_points']
                team_name = prediction['team_name']
                
                # Carte du joueur
                st.markdown(f'<div class="player-card">', unsafe_allow_html=True)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"### 👤 {player['web_name']}")
                    st.markdown(f"**🏟️ Équipe:** {team_name}")
                    st.markdown(f"**💰 Coût:** {player['cost']:.1f}M")
                    st.markdown(f"**📊 Position:** {st.session_state.predictor.get_position_name(player['element_type'])}")
                
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
                col3, col4, col5 = st.columns(3)
                
                with col3:
                    st.metric("Forme récente", f"{player['form']:.1f}")
                    st.metric("Points totaux", f"{player['total_points']}")
                
                with col4:
                    st.metric("Points par match", f"{player['points_per_game']:.1f}")
                    st.metric("Buts", f"{player['goals_scored']}")
                
                with col5:
                    st.metric("Passes", f"{player['assists']}")
                    st.metric("Minutes", f"{player['minutes']}")
                
                # Contexte du match
                st.markdown("---")
                st.markdown("#### 📅 Contexte du match")
                
                difficulty = player.get('next_opponent_difficulty', 3)
                home_away = "🏠 Domicile" if player.get('is_home', 0) == 1 else "✈️ Extérieur"
                difficulty_text = {1: "Très Facile 🟢", 2: "Facile 🟡", 3: "Moyen 🟠", 4: "Difficile 🔴", 5: "Très Difficile 🛑"}
                
                col6, col7 = st.columns(2)
                with col6:
                    st.info(f"**Lieu:** {home_away}")
                with col7:
                    st.info(f"**Difficulté:** {difficulty_text.get(difficulty, 'Moyen')}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("❌ Joueur non trouvé. Essaye avec un autre nom.")
    
    # Top 10 prédictions
    st.sidebar.markdown("---")
    if st.sidebar.button("🎯 Voir le Top 10"):
        st.session_state.show_top10 = True
    
    if st.session_state.get('show_top10', False):
        st.markdown("## 🏆 Top 10 des Prédictions")
        
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
                    'Joueur': player['web_name'],
                    'Équipe': st.session_state.predictor.get_team_name(player['team']),
                    'Position': st.session_state.predictor.get_position_name(player['element_type']),
                    'Points Prédits': final_prediction,
                    'Forme': player['form'],
                    'Coût': player['cost']
                })
        
        # Créer le dataframe et trier
        top_df = pd.DataFrame(all_predictions)
        top_df = top_df.nlargest(10, 'Points Prédits')
        
        # Afficher le tableau
        st.dataframe(top_df.style.format({
            'Points Prédits': '{:.1f}',
            'Forme': '{:.1f}',
            'Coût': '{:.1f}'
        }).background_gradient(subset=['Points Prédits'], cmap='YlOrRd'), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("### 💡 Comment utiliser:")
    st.markdown("""
    1. **🔍 Recherche** un joueur par son nom
    2. **🎯 Voir le Top 10** des meilleures prédictions
    3. **📊 Analyser** les statistiques et le contexte
    4. **🤝 Partager** avec tes amis!
    """)

if __name__ == "__main__":
    main()