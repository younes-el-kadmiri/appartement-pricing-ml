import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.preprocessing import Preprocessor
from src.modeling import ModelFactory

def test_model(path_data):
    print("Chargement et prétraitement des données...")
    preprocessor = Preprocessor()
    df = preprocessor.load_data(path_data)
    df = preprocessor.clean_and_transform(df)

    X, y = preprocessor.select_features(df)
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Données splittées : train={X_train.shape[0]}, test={X_test.shape[0]}")

    # Entraîner un modèle (par exemple LinearRegression)
    model = ModelFactory().get_models()['DecisionTree']
    model.fit(X_train, y_train)
    print("Modèle entraîné.")

    # Prédictions
    y_pred = model.predict(X_test)

    # Affichage quelques prédictions vs valeurs réelles
    print("\nExemples de prédictions (réel -> prédit):")
    for vrai, pred in zip(y_test[:10], y_pred[:10]):
        print(f"{vrai:.2f} -> {pred:.2f}")

    # Calcul métriques
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\nScores sur test set:")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R2   : {r2:.4f}")

    # Validation croisée
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"\nValidation croisée R2 (5 folds) : {np.mean(scores):.4f} ± {np.std(scores):.4f}")

if __name__ == "__main__":
    path = 'data/appartements_data_db.csv'  # adapte selon ton chemin
    test_model(path)
