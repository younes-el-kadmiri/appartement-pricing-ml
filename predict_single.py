import pandas as pd
import numpy as np
import joblib
from src.preprocessing import Preprocessor

# === Chargement du modèle ===
model_path = 'models/best_model_2.pkl'
model = joblib.load(model_path)

# === Exemple d'entrée utilisateur ===
example = {
    'city_name': 'Casablanca',
    'description': '',
    'salon': 1,
    'nb_rooms': 2,
    'nb_baths': 1,
    'surface_area': 120,
    'quarter': '',
    'latitude': None,
    'longitude': None
}
df_input = pd.DataFrame([example])

# === Initialisation du préprocesseur et transformation ===
preprocessor = joblib.load('models/preprocessor.pkl')

# Charger et transformer le jeu de données original pour obtenir les colonnes/features
df_base = preprocessor.load_data('data/appartements_data_db.csv')
df_cleaned = preprocessor.clean_and_transform(df_base)
X_train, y_train = preprocessor.select_features(df_cleaned)
selected_features = list(X_train.columns)

# Transformer les données d'entrée pour la prédiction
df_input_transformed = preprocessor.transform_for_prediction(df_input)

# Gérer les colonnes manquantes par rapport aux features d'entraînement
for col in selected_features:
    if col not in df_input_transformed.columns:
        df_input_transformed[col] = 0

# Réorganiser l'ordre des colonnes
X_input = df_input_transformed[selected_features]

# === Prédiction ===
log_pred = model.predict(X_input)[0]
predicted_price = round(float(np.exp(log_pred)), 2)

# === Affichage ===
print(f"✅ Prix prédit : {predicted_price*10000} MAD")
