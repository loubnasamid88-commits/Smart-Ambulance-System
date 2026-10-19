"""
=============================================================================
  ml_eta_module.py — Module ML avancé pour la prédiction ETA d'ambulances
  Système de Gestion d'Ambulances Intelligentes — Marrakech (PFE)
=============================================================================

UTILISATION DANS app.py :
  Remplacez le bloc "# ML — PRÉDICTION ETA" (lignes ~728-751) par :
      from ml_eta_module import build_and_train_model, predict_eta, ETAPredictor
      predictor = ETAPredictor()
      predictor.train_and_evaluate()          # affiche RMSE/MAE au démarrage
      predict_eta = predictor.predict         # API identique à l'ancienne fonction

=============================================================================
"""

import math
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

# XGBoost / LightGBM sont optionnels — le module se dégrade proprement s'ils manquent
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("⚠  XGBoost non installé  (pip install xgboost) — RandomForest utilisé")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("⚠  LightGBM non installé (pip install lightgbm) — RandomForest utilisé")


# ─────────────────────────────────────────────────────────────────────────────
# 1. FEATURE ENGINEERING
#    Nouvelles variables extraites / dérivées pour enrichir le dataset SUMO
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme le DataFrame brut en ajoutant des features riches.

    Colonnes attendues en entrée (minimales) :
        distance        — distance Haversine ambulance→accident (km)
        severity        — 0=low / 1=medium / 2=high
        hospital_load   — nombre de patients actuels à l'hôpital
        hour            — heure de la journée (0-23)

    Colonnes optionnelles (issues de SUMO via traci) :
        vehicle_density     — nb véhicules / km sur le trajet
        traffic_speed_ratio — vitesse_moyenne / vitesse_limite (0-1)
        nb_traffic_lights   — feux rouges sur le trajet
        road_type           — 0=résidentiel 1=secondaire 2=principale 3=autoroute
        day_of_week         — 0=lundi … 6=dimanche
        month               — 1-12

    Retourne le DataFrame enrichi prêt pour l'entraînement.
    """
    df = df.copy()

    # ── Temporel ────────────────────────────────────────────────────────────
    # Encodage cyclique de l'heure (évite la rupture 23h → 0h)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Indicateurs de pic de trafic (Marrakech : 8h-10h, 12h-14h, 17h-20h)
    df["is_rush_hour"] = df["hour"].apply(
        lambda h: 1 if (8 <= h <= 10 or 12 <= h <= 14 or 17 <= h <= 20) else 0
    )
    # Nuit (trafic fluide mais vitesse parfois + élevée)
    df["is_night"] = df["hour"].apply(lambda h: 1 if (22 <= h or h <= 5) else 0)

    if "day_of_week" in df.columns:
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        # Haute saison touristique (impact sur le trafic à Marrakech)
        df["is_high_season"] = df["month"].apply(
            lambda m: 1 if m in [3, 4, 5, 10, 11] else 0
        )

    # ── Distance ────────────────────────────────────────────────────────────
    # Terme quadratique (la relation distance→temps n'est pas purement linéaire)
    df["distance_sq"]   = df["distance"] ** 2
    df["distance_log"]  = np.log1p(df["distance"])

    # ── Trafic (si données SUMO disponibles) ────────────────────────────────
    if "vehicle_density" in df.columns:
        # Congestion estimée : densité normalisée × facteur rush
        df["congestion_index"] = (
            df["vehicle_density"] * (1 + df.get("is_rush_hour", 0) * 0.5)
        )
        df["density_x_dist"] = df["vehicle_density"] * df["distance"]

    if "traffic_speed_ratio" in df.columns:
        # Plus ce ratio est bas, plus le trafic est congestionné
        df["speed_ratio_inv"] = 1.0 / (df["traffic_speed_ratio"].clip(lower=0.1))

    if "nb_traffic_lights" in df.columns:
        # Temps moyen estimé perdu par feu rouge (≈ 30 s en urgence)
        df["traffic_light_delay"] = df["nb_traffic_lights"] * 0.5  # minutes

    # ── Type de route ────────────────────────────────────────────────────────
    if "road_type" in df.columns:
        # One-hot (0=résidentiel, 1=secondaire, 2=principale, 3=autoroute)
        for rt in [0, 1, 2, 3]:
            df[f"road_{rt}"] = (df["road_type"] == rt).astype(int)

    # ── Hôpital ──────────────────────────────────────────────────────────────
    df["hospital_load_ratio"] = df["hospital_load"] / 100.0   # normalisation douce
    df["hosp_overloaded"]     = (df["hospital_load"] > 75).astype(int)

    # ── Sévérité × distance (interaction) ───────────────────────────────────
    df["severity_x_distance"] = df["severity"] * df["distance"]
    df["severity_x_rush"]     = df["severity"] * df.get("is_rush_hour", 0)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Retourne la liste des features disponibles après feature engineering."""
    # Features de base toujours présentes
    base = [
        "distance", "distance_sq", "distance_log",
        "severity", "severity_x_distance",
        "hospital_load", "hospital_load_ratio", "hosp_overloaded",
        "hour", "hour_sin", "hour_cos", "is_rush_hour", "is_night",
    ]
    # Features optionnelles (SUMO enrichi)
    optional = [
        "vehicle_density", "congestion_index", "density_x_dist",
        "traffic_speed_ratio", "speed_ratio_inv",
        "nb_traffic_lights", "traffic_light_delay",
        "road_type", "road_0", "road_1", "road_2", "road_3",
        "day_of_week", "dow_sin", "dow_cos", "is_weekend",
        "month", "month_sin", "month_cos", "is_high_season",
        "severity_x_rush",
    ]
    return [c for c in base + optional if c in df.columns]


# ─────────────────────────────────────────────────────────────────────────────
# 2. GÉNÉRATION DES DONNÉES DE SIMULATION
#    Reproduit la logique de votre _build_ml_model() mais enrichie
# ─────────────────────────────────────────────────────────────────────────────

def generate_training_data(n_samples: int = 3000, include_sumo_features: bool = True) -> pd.DataFrame:
    """
    Génère un dataset d'entraînement synthétique calé sur les conditions
    réelles de Marrakech (avec variables SUMO simulées).

    Pour un vrai PFE, remplacez cette fonction par l'import de vos logs SUMO :
        df = pd.read_csv("sumo_logs.csv")
    """
    rows = []
    for _ in range(n_samples):
        # ── Features de base ────────────────────────────────────────────────
        distance   = random.uniform(0.5, 15.0)    # km
        severity   = random.choice([0, 1, 2])      # low/medium/high
        hosp_load  = random.randint(5, 95)
        hour       = random.randint(0, 23)
        dow        = random.randint(0, 6)          # 0=lundi
        month      = random.randint(1, 12)

        # ── Features SUMO (simulées — à remplacer par traci) ────────────────
        is_rush = 1 if (8 <= hour <= 10 or 12 <= hour <= 14 or 17 <= hour <= 20) else 0
        is_night = 1 if (22 <= hour or hour <= 5) else 0

        # Densité de véhicules : plus forte en rush, en semaine, haute saison
        base_density = random.gauss(35, 15)
        density = max(0, base_density * (1.7 if is_rush else 1.0)
                                       * (0.6 if is_night else 1.0)
                                       * (1.3 if month in [3,4,5,10,11] else 1.0))

        # Ratio vitesse : baisse si densité forte
        speed_ratio = max(0.1, min(1.0, 1.0 - density / 200.0 + random.gauss(0, 0.05)))

        # Feux rouges : proportionnel à la distance, + nombreux centre-ville
        nb_lights = max(0, int(distance * random.gauss(2.5, 0.8)))

        # Type de route dominant
        road_type = random.choices([0, 1, 2, 3], weights=[20, 35, 35, 10])[0]

        # ── ETA réaliste (label) ─────────────────────────────────────────────
        # Vitesse de base selon type de route (km/h)
        base_speed = {0: 25, 1: 40, 2: 55, 3: 80}[road_type]
        # Ambulance peut forcer son passage mais pas à vitesse illimitée
        effective_speed = base_speed * speed_ratio * random.uniform(1.1, 1.4)
        effective_speed = max(10, min(effective_speed, 80))

        # Temps de trajet de base
        base_time = (distance / effective_speed) * 60  # minutes

        # Délai feux rouges (l'ambulance passe mais ralentit)
        light_delay = nb_lights * random.uniform(0.1, 0.4)

        # Bonus sévérité : high → ambulanciers plus pressés
        sev_factor = {0: 1.05, 1: 1.0, 2: 0.92}[severity]

        # Pénalité charge hôpital (embouteillage au drop-off)
        hosp_penalty = (hosp_load / 100.0) * random.uniform(0.5, 2.0)

        eta = round(
            base_time * sev_factor + light_delay + hosp_penalty
            + random.gauss(0, 0.3),  # bruit gaussien
            2
        )
        eta = max(0.5, eta)  # minimum 30 secondes

        rows.append({
            "distance":           round(distance, 3),
            "severity":           severity,
            "hospital_load":      hosp_load,
            "hour":               hour,
            "day_of_week":        dow,
            "month":              month,
            "vehicle_density":    round(max(0, density), 1),
            "traffic_speed_ratio": round(speed_ratio, 3),
            "nb_traffic_lights":  nb_lights,
            "road_type":          road_type,
            "eta":                eta,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3. OPTIMISATION DES HYPERPARAMÈTRES
# ─────────────────────────────────────────────────────────────────────────────

def optimize_random_forest(X_train, y_train, n_iter: int = 30, cv: int = 5) -> dict:
    """
    RandomizedSearchCV pour RandomForestRegressor.
    Retourne les meilleurs hyperparamètres trouvés.
    """
    print("\n🔍 Optimisation RandomForest (RandomizedSearchCV)…")
    param_dist = {
        "n_estimators":      [100, 200, 300, 400, 500],
        "max_depth":         [None, 8, 12, 16, 20, 25],
        "min_samples_split": [2, 4, 6, 8, 10],
        "min_samples_leaf":  [1, 2, 3, 4],
        "max_features":      ["sqrt", "log2", 0.5, 0.7],
        "bootstrap":         [True, False],
    }
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        rf, param_dist,
        n_iter=n_iter, cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"  ✅ Meilleur RMSE CV  : {-search.best_score_:.3f} min")
    print(f"  📋 Meilleurs params  : {search.best_params_}")
    return search.best_params_


def optimize_xgboost(X_train, y_train, n_iter: int = 30, cv: int = 5) -> dict:
    """RandomizedSearchCV pour XGBoost."""
    if not XGBOOST_AVAILABLE:
        return {}
    print("\n🔍 Optimisation XGBoost (RandomizedSearchCV)…")
    param_dist = {
        "n_estimators":     [200, 300, 400, 500],
        "max_depth":        [3, 4, 5, 6, 7, 8],
        "learning_rate":    [0.01, 0.03, 0.05, 0.1, 0.15],
        "subsample":        [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma":            [0, 0.1, 0.2, 0.5],
        "reg_alpha":        [0, 0.01, 0.1, 1.0],
        "reg_lambda":       [0.5, 1.0, 1.5, 2.0],
    }
    xgb = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    search = RandomizedSearchCV(
        xgb, param_dist,
        n_iter=n_iter, cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"  ✅ Meilleur RMSE CV  : {-search.best_score_:.3f} min")
    print(f"  📋 Meilleurs params  : {search.best_params_}")
    return search.best_params_


# ─────────────────────────────────────────────────────────────────────────────
# 4. PRÉTRAITEMENT & PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_preprocessing_pipeline(model):
    """
    Encapsule le modèle dans un Pipeline sklearn avec StandardScaler.

    Pour RandomForest/XGBoost/LightGBM, la normalisation n'est pas strictement
    nécessaire (modèles basés sur des arbres), mais elle :
      - facilite la comparaison des feature importances
      - accélère la convergence si on ajoute un réseau neuronal plus tard
      - garantit la cohérence si on passe à un SVM ou régression linéaire
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  model),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 5. ÉVALUATION — RMSE, MAE, R²
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str = "Modèle") -> dict:
    """
    Calcule et affiche RMSE, MAE, R² sur le jeu de test.
    Affiché dans la console Flask au démarrage (visible dans le terminal).
    """
    y_pred = model.predict(X_test)
    rmse   = math.sqrt(mean_squared_error(y_test, y_pred))
    mae    = mean_absolute_error(y_test, y_pred)
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Erreur relative moyenne (en %)
    mape   = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 0.01))) * 100

    print(f"\n{'='*55}")
    print(f"  📊 ÉVALUATION ML — {model_name}")
    print(f"{'='*55}")
    print(f"  RMSE  : {rmse:.3f} min  (erreur quadratique moyenne)")
    print(f"  MAE   : {mae:.3f} min  (erreur absolue moyenne)")
    print(f"  MAPE  : {mape:.1f}%    (erreur relative moyenne)")
    print(f"  R²    : {r2:.4f}      (variance expliquée)")
    print(f"{'='*55}\n")

    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def print_feature_importance(model, feature_names: list, top_n: int = 10):
    """Affiche les features les plus importantes dans la console."""
    # Compatibilité Pipeline / modèle nu
    estimator = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model

    if not hasattr(estimator, "feature_importances_"):
        return

    importances = estimator.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print(f"\n  🔑 TOP {top_n} FEATURES LES PLUS IMPORTANTES :")
    for i, (name, imp) in enumerate(pairs[:top_n], 1):
        bar = "█" * int(imp * 60)
        print(f"  {i:2d}. {name:<30s} {imp:.4f}  {bar}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLASSE PRINCIPALE — ETAPredictor
# ─────────────────────────────────────────────────────────────────────────────

class ETAPredictor:
    """
    Classe principale qui gère l'entraînement, l'évaluation et la prédiction.

    Utilisation dans app.py :
        predictor = ETAPredictor(model_type="xgboost")   # ou "rf", "lgbm"
        predictor.train_and_evaluate()
        eta = predictor.predict(distance=3.2, severity_str="high", hospital_load=45)
    """

    def __init__(
        self,
        model_type: str = "auto",   # "auto" choisit le meilleur disponible
        n_samples:  int = 3000,
        optimize:   bool = False,   # True = lance RandomizedSearchCV (plus lent)
        n_iter:     int = 20,       # itérations pour RandomizedSearchCV
    ):
        # Choix du modèle
        if model_type == "auto":
            if XGBOOST_AVAILABLE:
                model_type = "xgboost"
            elif LIGHTGBM_AVAILABLE:
                model_type = "lgbm"
            else:
                model_type = "rf"

        self.model_type  = model_type
        self.n_samples   = n_samples
        self.optimize    = optimize
        self.n_iter      = n_iter
        self.model       = None
        self.feature_cols: list = []
        self.metrics: dict = {}

    # ── Entraînement ─────────────────────────────────────────────────────────
    def train_and_evaluate(self, df: pd.DataFrame = None) -> dict:
        """
        Génère (ou utilise) les données, fait le feature engineering,
        entraîne le modèle choisi et évalue ses performances.

        Paramètre df : si None, génère des données synthétiques.
                       Si fourni, doit contenir les colonnes minimales.
        """
        print(f"\n🚀 Démarrage entraînement ML — modèle : {self.model_type.upper()}")

        # 1. Données
        if df is None:
            print(f"  📦 Génération de {self.n_samples} exemples synthétiques…")
            df = generate_training_data(self.n_samples)

        # 2. Feature engineering
        df_eng = engineer_features(df)
        self.feature_cols = get_feature_columns(df_eng)

        X = df_eng[self.feature_cols].values
        y = df_eng["eta"].values

        # 3. Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 4. Construction du modèle
        base_model = self._build_base_model(X_train, y_train)
        self.model = build_preprocessing_pipeline(base_model)

        # 5. Entraînement
        self.model.fit(X_train, y_train)

        # 6. Évaluation
        model_label = f"{self.model_type.upper()} ({len(self.feature_cols)} features)"
        self.metrics = evaluate_model(self.model, X_test, y_test, model_label)

        # 7. Importance des features
        print_feature_importance(self.model, self.feature_cols)

        print(f"✅ Modèle ML ({self.model_type.upper()}) prêt — "
              f"RMSE={self.metrics['rmse']:.2f} min | MAE={self.metrics['mae']:.2f} min\n")

        return self.metrics

    def _build_base_model(self, X_train, y_train):
        """Instancie et optionnellement optimise le modèle de base."""
        if self.model_type == "xgboost" and XGBOOST_AVAILABLE:
            if self.optimize:
                best = optimize_xgboost(X_train, y_train, n_iter=self.n_iter)
                return XGBRegressor(random_state=42, n_jobs=-1, verbosity=0, **best)
            # Hyperparamètres déjà bons pour ce type de données
            return XGBRegressor(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )

        if self.model_type == "lgbm" and LIGHTGBM_AVAILABLE:
            return lgb.LGBMRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )

        # Fallback : RandomForest
        if self.optimize:
            best = optimize_random_forest(X_train, y_train, n_iter=self.n_iter)
            return RandomForestRegressor(random_state=42, n_jobs=-1, **best)
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=16,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )

    # ── Prédiction — API identique à l'ancienne predict_eta() ────────────────
    def predict(
        self,
        distance: float,
        severity_str: str,
        hospital_load: int,
        # Variables SUMO enrichies (optionnelles)
        vehicle_density: float    = None,
        traffic_speed_ratio: float = None,
        nb_traffic_lights: int    = None,
        road_type: int            = None,
    ) -> float:
        """
        Prédit l'ETA en minutes.

        Paramètres obligatoires (compatibles avec l'ancienne API) :
            distance       — km (Haversine)
            severity_str   — "low" | "medium" | "high"
            hospital_load  — entier 0-100

        Paramètres optionnels (SUMO) :
            vehicle_density      — véhicules/km
            traffic_speed_ratio  — 0.0-1.0
            nb_traffic_lights    — entier
            road_type            — 0/1/2/3

        Retourne l'ETA arrondi à 2 décimales (en minutes).
        """
        if self.model is None:
            raise RuntimeError("Appelez train_and_evaluate() avant predict().")

        now   = datetime.now()
        hour  = now.hour
        dow   = now.weekday()
        month = now.month
        sev   = {"low": 0, "medium": 1, "high": 2}.get(severity_str, 1)

        # Construire une ligne avec toutes les colonnes possibles
        row = {
            "distance":           distance,
            "severity":           sev,
            "hospital_load":      hospital_load,
            "hour":               hour,
            "day_of_week":        dow,
            "month":              month,
        }
        if vehicle_density    is not None: row["vehicle_density"]    = vehicle_density
        if traffic_speed_ratio is not None: row["traffic_speed_ratio"] = traffic_speed_ratio
        if nb_traffic_lights  is not None: row["nb_traffic_lights"]  = nb_traffic_lights
        if road_type          is not None: row["road_type"]           = road_type

        df_row = engineer_features(pd.DataFrame([row]))

        # N'utiliser que les colonnes sur lesquelles le modèle a été entraîné
        for col in self.feature_cols:
            if col not in df_row.columns:
                df_row[col] = 0.0  # valeur neutre par défaut

        X = df_row[self.feature_cols].values
        eta = float(self.model.predict(X)[0])
        return round(max(0.1, eta), 2)

    # ── Méthode utilitaire : ré-entraîner depuis des logs SUMO ───────────────
    def retrain_from_logs(self, csv_path: str):
        """
        Charge un fichier CSV de logs SUMO et ré-entraîne le modèle.
        Colonnes attendues : voir generate_training_data().
        """
        print(f"📂 Chargement logs SUMO : {csv_path}")
        df = pd.read_csv(csv_path)
        return self.train_and_evaluate(df)


# ─────────────────────────────────────────────────────────────────────────────
# 7. FONCTIONS DE COMMODITÉ (compatibilité directe avec app.py)
# ─────────────────────────────────────────────────────────────────────────────

# Instance globale — créée une seule fois au import
_predictor: ETAPredictor = None


def build_and_train_model(
    model_type: str = "auto",
    n_samples:  int = 3000,
    optimize:   bool = False,
) -> ETAPredictor:
    """
    Initialise et entraîne le predictor global.
    À appeler une fois dans app.py au démarrage :

        from ml_eta_module import build_and_train_model, predict_eta
        build_and_train_model()     # entraîne et affiche les métriques
    """
    global _predictor
    _predictor = ETAPredictor(model_type=model_type, n_samples=n_samples, optimize=optimize)
    _predictor.train_and_evaluate()
    return _predictor


def predict_eta(distance: float, severity_str: str, hospital_load: int, **kwargs) -> float:
    """
    Wrapper compatible avec l'ancienne signature de predict_eta() dans app.py.
    Appelable sans changer le reste du code.
    """
    global _predictor
    if _predictor is None:
        # Initialisation de secours si oublié
        build_and_train_model()
    return _predictor.predict(distance, severity_str, hospital_load, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 8. SCRIPT AUTONOME — test hors Flask
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  TEST AUTONOME — ml_eta_module.py")
    print("=" * 60)

    # ── Comparaison des 3 modèles ────────────────────────────────────────────
    results = {}
    for mtype in ["rf", "xgboost", "lgbm"]:
        if mtype == "xgboost" and not XGBOOST_AVAILABLE:
            continue
        if mtype == "lgbm" and not LIGHTGBM_AVAILABLE:
            continue
        p = ETAPredictor(model_type=mtype, n_samples=3000)
        metrics = p.train_and_evaluate()
        results[mtype] = metrics

    # ── Tableau récapitulatif ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  📊 COMPARAISON DES MODÈLES")
    print(f"  {'Modèle':<12} {'RMSE':>8} {'MAE':>8} {'MAPE':>8} {'R²':>8}")
    print("  " + "-" * 50)
    for name, m in results.items():
        print(f"  {name.upper():<12} {m['rmse']:>7.3f}m {m['mae']:>7.3f}m "
              f"{m['mape']:>7.1f}% {m['r2']:>7.4f}")
    print("=" * 55)

    # ── Exemple de prédiction ────────────────────────────────────────────────
    print("\n  🚑 Exemple de prédiction (distance=4.5km, high, load=60) :")
    for name, mtype in [("RandomForest", "rf"), ("XGBoost", "xgboost"), ("LightGBM", "lgbm")]:
        if mtype == "xgboost" and not XGBOOST_AVAILABLE: continue
        if mtype == "lgbm"    and not LIGHTGBM_AVAILABLE: continue
        p = ETAPredictor(model_type=mtype, n_samples=3000)
        p.train_and_evaluate()
        eta = p.predict(4.5, "high", 60, vehicle_density=45, nb_traffic_lights=8)
        print(f"  {name:<15}: ETA = {eta:.2f} min")
