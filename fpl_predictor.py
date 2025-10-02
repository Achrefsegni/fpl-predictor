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

class FPLPlayerPredictor:
    def __init__(self):
        self.players_df = None
        self.teams_df = None
        self.fixtures_df = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def load_all_fpl_data(self):
        """Charge TOUTES les données disponibles de l'API FPL"""
        print("📥 CHARGEMENT DES DONNÉES FPL...")
        
        bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        response = requests.get(bootstrap_url)
        data = response.json()
        
        self.players_df = pd.DataFrame(data['elements'])
        self.teams_df = pd.DataFrame(data['teams'])
        
        fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"
        fixtures_response = requests.get(fixtures_url)
        self.fixtures_df = pd.DataFrame(fixtures_response.json())
        
        print(f"✅ Données chargées: {len(self.players_df)} joueurs")
        
    def clean_numeric_data(self, series):
        """Nettoie une série de données pour la convertir en numérique"""
        if pd.api.types.is_numeric_dtype(series):
            return series
        
        series_clean = series.replace([None, 'None', 'null', ''], '0')
        
        try:
            return pd.to_numeric(series_clean, errors='coerce')
        except:
            return pd.Series([0] * len(series), index=series.index)
    
    def create_realistic_features(self):
        """Crée des features réalistes pour prédire les points par GW"""
        print("🧮 CRÉATION DES FEATURES RÉALISTES...")
        
        # NETTOYAGE DES DONNÉES DE BASE
        critical_columns = ['form', 'points_per_game', 'goals_scored', 'assists', 'minutes', 
                           'total_points', 'influence', 'creativity', 'threat', 'ict_index', 'bps']
        
        for col in critical_columns:
            if col in self.players_df.columns:
                self.players_df[col] = self.clean_numeric_data(self.players_df[col])
        
        # 1. FEATURES PRINCIPALES RÉALISTES
        self.players_df['cost'] = self.players_df['now_cost'] / 10
        
        # Points par minute (très important)
        self.players_df['points_per_minute'] = self.players_df['total_points'] / self.players_df['minutes']
        self.players_df['points_per_minute'] = self.players_df['points_per_minute'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # 2. IMPACT OFFENSIF RÉALISTE
        self.players_df['goal_involvement_per_90'] = (self.players_df['goals_scored'] + self.players_df['assists']) / (self.players_df['minutes'] / 90)
        self.players_df['goal_involvement_per_90'] = self.players_df['goal_involvement_per_90'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # 3. CONSISTANCE (le facteur le plus important!)
        self.players_df['consistency_score'] = self.players_df['points_per_game'] / self.players_df['points_per_game'].max()
        
        # 4. FORM RÉCENTE vs MOYENNE
        self.players_df['form_vs_average'] = self.players_df['form'] / self.players_df['points_per_game']
        self.players_df['form_vs_average'] = self.players_df['form_vs_average'].replace([np.inf, -np.inf], 1).fillna(1)
        
        # 5. MINUTES STABILITY (fiabilité)
        self.players_df['minutes_ratio'] = self.players_df['minutes'] / (6 * 90)  # 6 GW × 90 minutes
        self.players_df['minutes_ratio'] = self.players_df['minutes_ratio'].clip(0, 1)
        
        # 6. FEATURES D'EQUIPE
        team_strength = dict(zip(self.teams_df['id'], self.teams_df['strength']))
        team_attack = dict(zip(self.teams_df['id'], self.teams_df['strength_attack_home']))  # Approximation
        
        self.players_df['team_strength'] = self.players_df['team'].map(team_strength)
        self.players_df['team_attack'] = self.players_df['team'].map(team_attack)
        
        # 7. CONTEXTE DU PROCHAIN MATCH (GW7)
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
        
        # Facteur de difficulté (1=facile, 5=difficile → inversé)
        self.players_df['difficulty_factor'] = (6 - self.players_df['next_opponent_difficulty']) / 5
        
        print(f"✅ Features réalistes créées")
    
    def create_realistic_target(self):
        """
        Crée une target RÉALISTE pour l'entraînement
        On utilise points_per_game comme approximation des points par GW
        """
        print("🎯 CRÉATION DE LA TARGET RÉALISTE...")
        
        # La target idéale serait les points par GW historiques
        # Mais comme on ne les a pas, on utilise points_per_game
        target = self.players_df['points_per_game'].copy()
        
        # Ajustement pour les joueurs avec peu de minutes
        minutes_threshold = 180  # Au moins 2 matchs complets
        low_minutes_players = self.players_df['minutes'] < minutes_threshold
        
        # Pour les joueurs avec peu de minutes, on réduit la confiance
        target[low_minutes_players] = target[low_minutes_players] * 0.7
        
        # Limiter les prédictions à un maximum réaliste
        # Très peu de joueurs dépassent 15 points par GW
        target = target.clip(upper=12.0)
        
        print(f"📊 Target - Moyenne: {target.mean():.2f}, Max: {target.max():.2f}")
        
        return target
    
    def prepare_features(self):
        """Prépare les features pour le modèle"""
        
        # Features sélectionnées pour leur importance
        feature_columns = [
            # FACTEURS PRINCIPAUX (80% de l'importance)
            'points_per_game',           # 🥇 Le plus important - consistance
            'form',                      # 🥈 Forme récente
            'points_per_minute',         # 🥉 Efficacité
            'consistency_score',         # Stabilité
            
            # FACTEURS SECONDAIRES (15%)
            'goal_involvement_per_90',   # Impact offensif
            'minutes_ratio',             # Fiabilité temps de jeu
            'team_attack',               # Force offensive équipe
            'cost',                      # Qualité du joueur
            
            # FACTEURS CONTEXTUELS (5%)
            'difficulty_factor',         # Difficulté adversaire
            'is_home',                   # Avantage domicile
            'form_vs_average',           # Élan récent
        ]
        
        # Filtrer les colonnes disponibles
        available_features = [col for col in feature_columns if col in self.players_df.columns]
        print(f"🎯 {len(available_features)} features réalistes sélectionnées")
        
        # Préparer les données
        X = self.players_df[available_features].copy()
        
        # Nettoyage final
        for col in X.columns:
            X[col] = self.clean_numeric_data(X[col])
            X[col] = X[col].fillna(0)
            X[col] = X[col].replace([np.inf, -np.inf], 0)
        
        return X, available_features
    
    def train_realistic_model(self):
        """Entraîne un modèle réaliste"""
        print("🤖 ENTRAÎNEMENT DU MODÈLE RÉALISTE...")
        
        X, feature_names = self.prepare_features()
        y = self.create_realistic_target()
        
        # Filtrer les joueurs avec des minutes raisonnables
        valid_players = self.players_df['minutes'] > 90
        X_filtered = X[valid_players]
        y_filtered = y[valid_players]
        
        if len(X_filtered) == 0:
            print("❌ Aucun joueur valide pour l'entraînement")
            return 0, 0
        
        print(f"📊 Données d'entraînement: {X_filtered.shape[0]} joueurs")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_filtered, test_size=0.2, random_state=42
        )
        
        # Modèle XGBoost optimisé pour des prédictions réalistes
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,           # Plus shallow pour éviter overfitting
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.2,         # Plus de régularisation
            reg_lambda=0.2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Évaluation
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"📊 Performance du modèle réaliste:")
        print(f"   MAE: {mae:.2f} points par GW")
        print(f"   RMSE: {rmse:.2f} points par GW")
        
        # Importance des features
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return mae, rmse

    def search_player(self, player_name):
        """
        Recherche un joueur par nom (recherche partielle)
        Retourne les joueurs correspondants
        """
        player_name_lower = player_name.lower()
        
        # Recherche dans les noms complets
        matches = self.players_df[
            self.players_df['web_name'].str.lower().str.contains(player_name_lower, na=False)
        ]
        
        if len(matches) == 0:
            # Recherche dans les prénoms + noms
            matches = self.players_df[
                (self.players_df['first_name'].str.lower().str.contains(player_name_lower, na=False)) |
                (self.players_df['second_name'].str.lower().str.contains(player_name_lower, na=False))
            ]
        
        return matches

    def predict_player_points(self, player_name):
        """
        Prédit les points pour un joueur spécifique
        """
        print(f"\n🔍 RECHERCHE DU JOUEUR: {player_name}")
        
        # Rechercher le joueur
        matches = self.search_player(player_name)
        
        if len(matches) == 0:
            print(f"❌ Aucun joueur trouvé avec le nom '{player_name}'")
            return None
        
        if len(matches) > 1:
            print(f"🔍 Plusieurs joueurs trouvés:")
            for i, (_, player) in enumerate(matches.head(5).iterrows()):
                print(f"   {i+1}. {player['web_name']} ({self.get_team_name(player['team'])})")
            
            if len(matches) > 5:
                print(f"   ... et {len(matches) - 5} autres")
            
            print("💡 Sois plus spécifique dans ta recherche!")
            return None
        
        # Un seul joueur trouvé
        player = matches.iloc[0]
        return self.predict_single_player(player)

    def predict_single_player(self, player):
        """
        Prédit les points pour un joueur spécifique
        """
        # Préparer les features pour la prédiction
        X, feature_names = self.prepare_features()
        
        # Trouver l'index du joueur
        player_mask = self.players_df['id'] == player['id']
        if not player_mask.any():
            print(f"❌ Joueur non trouvé dans les données")
            return None
        
        # Features du joueur
        player_features = X[player_mask].iloc[0].values.reshape(1, -1)
        
        # Prédiction
        predicted_points = self.model.predict(player_features)[0]
        
        # AJUSTEMENTS RÉALISTES
        minutes_factor = player['minutes_ratio'] if 'minutes_ratio' in player else 1
        difficulty_factor = player['difficulty_factor'] if 'difficulty_factor' in player else 1
        
        # Appliquer les ajustements
        final_prediction = predicted_points * minutes_factor * difficulty_factor
        final_prediction = np.clip(final_prediction, 0, 12)  # Limite réaliste
        
        return {
            'player': player,
            'predicted_points': final_prediction,
            'base_prediction': predicted_points,
            'team_name': self.get_team_name(player['team'])
        }

    def get_team_name(self, team_id):
        """Retourne le nom de l'équipe"""
        team_mapping = dict(zip(self.teams_df['id'], self.teams_df['name']))
        return team_mapping.get(team_id, 'Inconnu')

    def display_player_prediction(self, prediction_result):
        """
        Affiche la prédiction de façon élégante
        """
        if prediction_result is None:
            return
        
        player = prediction_result['player']
        predicted_points = prediction_result['predicted_points']
        base_prediction = prediction_result['base_prediction']
        team_name = prediction_result['team_name']
        
        print("\n" + "="*60)
        print("🎯 PRÉDICTION DE POINTS - GW7")
        print("="*60)
        
        # Informations du joueur
        print(f"👤 JOUEUR: {player['web_name']}")
        print(f"🏟️  ÉQUIPE: {team_name}")
        print(f"💰 COÛT: {player['cost'] if 'cost' in player else player['now_cost']/10:.1f}M")
        print(f"📊 POSITION: {self.get_position_name(player['element_type'])}")
        
        print("\n" + "─"*40)
        
        # Statistiques actuelles
        print("📈 STATISTIQUES ACTUELLES:")
        print(f"   • Forme récente: {player['form']:.1f}")
        print(f"   • Points par match: {player['points_per_game']:.1f}")
        print(f"   • Points totaux: {player['total_points']}")
        print(f"   • Buts: {player['goals_scored']} | Passes: {player['assists']}")
        print(f"   • Minutes jouées: {player['minutes']}")
        
        print("\n" + "─"*40)
        
        # Contexte du match
        print("📅 CONTEXTE DU MATCH:")
        difficulty = player.get('next_opponent_difficulty', 3)
        home_away = "🏠 DOMICILE" if player.get('is_home', 0) == 1 else "✈️ EXTÉRIEUR"
        difficulty_text = {1: "Très Facile 🟢", 2: "Facile 🟡", 3: "Moyen 🟠", 4: "Difficile 🔴", 5: "Très Difficile 🛑"}
        
        print(f"   • {home_away}")
        print(f"   • Difficulté: {difficulty_text.get(difficulty, 'Moyen')}")
        
        print("\n" + "─"*40)
        
        # PRÉDICTION
        print("🔮 PRÉDICTION POUR LA GW7:")
        
        # Afficher avec emoji selon le score
        if predicted_points >= 8:
            emoji = "🔥🔥🔥"
            color = "\033[92m"  # Vert
        elif predicted_points >= 6:
            emoji = "🔥🔥"
            color = "\033[93m"  # Jaune
        elif predicted_points >= 4:
            emoji = "🔥"
            color = "\033[96m"  # Cyan
        else:
            emoji = "⚡"
            color = "\033[94m"  # Bleu
        
        reset_color = "\033[0m"
        
        print(f"   {color}{emoji} {predicted_points:.1f} POINTS PRÉDITS {emoji}{reset_color}")
        
        # Interprétation
        print(f"\n💡 INTERPRÉTATION:")
        if predicted_points >= 8:
            print("   Excellent choix! Fort potentiel de points cette semaine.")
        elif predicted_points >= 6:
            print("   Bon choix! Performance solide attendue.")
        elif predicted_points >= 4:
            print("   Choix décent. Performance moyenne attendue.")
        else:
            print("   Choix risqué. Performance limitée attendue.")
        
        print("="*60)

    def get_position_name(self, position_code):
        """Retourne le nom de la position"""
        positions = {1: "Gardien 🧤", 2: "Défenseur 🛡️", 3: "Milieu 🎯", 4: "Attaquant ⚽"}
        return positions.get(position_code, "Inconnu")

    def interactive_search(self):
        """
        Mode interactif pour rechercher des joueurs
        """
        print("\n" + "🎮 MODE RECHERCHE INTERACTIF")
        print("="*50)
        print("Tape le nom d'un joueur (ou 'quit' pour quitter)")
        print("Exemples: 'Salah', 'Haaland', 'Kane'")
        print("="*50)
        
        while True:
            try:
                player_input = input("\n🔍 Entrez le nom d'un joueur: ").strip()
                
                if player_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 À bientôt!")
                    break
                
                if not player_input:
                    continue
                
                # Faire la prédiction
                prediction = self.predict_player_points(player_input)
                
                if prediction:
                    self.display_player_prediction(prediction)
                
            except KeyboardInterrupt:
                print("\n👋 À bientôt!")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")

# EXÉCUTION PRINCIPALE
def main():
    # Initialiser le prédicteur
    predictor = FPLPlayerPredictor()
    
    try:
        print("🚀 INITIALISATION DU PRÉDICTEUR FPL...")
        
        # 1. Charger les données
        predictor.load_all_fpl_data()
        
        # 2. Créer les features réalistes
        predictor.create_realistic_features()
        
        # 3. Entraîner le modèle
        mae, rmse = predictor.train_realistic_model()
        
        print(f"\n✅ MODÈLE PRÊT! Performance: MAE={mae:.2f}, RMSE={rmse:.2f}")
        
        # 4. Lancer la recherche interactive
        predictor.interactive_search()
        
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution: {e}")
        import traceback
        traceback.print_exc()

# EXÉCUTION RAPIDE POUR TESTER UN JOUEUR
def quick_test():
    """Test rapide d'un joueur spécifique"""
    predictor = FPLPlayerPredictor()
    
    # Charger et entraîner rapidement
    predictor.load_all_fpl_data()
    predictor.create_realistic_features()
    predictor.train_realistic_model()
    
    # Tester avec un joueur spécifique
    test_players = ["Salah", "Haaland", "Kane", "Son"]
    
    for player in test_players:
        prediction = predictor.predict_player_points(player)
        if prediction:
            predictor.display_player_prediction(prediction)

if __name__ == "__main__":
    main()
    
    # Décommentez pour tester rapidement:
    # quick_test()