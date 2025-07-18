# src/evaluation.py

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, X_test, y_test):
        """
        Calcule les métriques d’évaluation pour le modèle fourni.

        Paramètres :
        - X_test : les features de test
        - y_test : les vraies valeurs

        Retourne :
        - Un dictionnaire avec MSE, RMSE, MAE et R2
        """
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

    @staticmethod
    def print_scores(scores):
        """
        Affiche joliment les scores d’évaluation.
        """
        print("===== Évaluation du modèle =====")
        print(f"MSE  : {scores['MSE']:.2f}")
        print(f"RMSE : {scores['RMSE']:.2f}")
        print(f"MAE  : {scores['MAE']:.2f}")
        print(f"R²   : {scores['R2']:.4f}")
