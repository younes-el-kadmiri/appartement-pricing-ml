import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.preprocessing import Preprocessor
from src.modeling import ModelTrainer
from src.evaluation import Evaluator

# Étape 1 : Chargement et préparation des données
preprocessor = Preprocessor()
df = preprocessor.load_data('data/appartements_data_db.csv')
print("✅ Données chargées")
df = preprocessor.clean_and_transform(df)
print("✅ Données nettoyées et transformées")
X, y = preprocessor.select_features(df)
print(f"✅ Features sélectionnées : {X.shape}, Target : {y.shape}")
# Séparation train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Étape 2 : Définition des modèles
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'RandomForest': RandomForestRegressor(random_state=42),
    'ExtraTrees': ExtraTreesRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'SVR': SVR(),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

# Étape 3 : Entraînement, évaluation et détection du sur-apprentissage
best_models = []

for name, model in models.items():
    print(f"\n🔧 Entraînement du modèle : {name}")
    trainer = ModelTrainer(model)
    trained_model = trainer.train(X_train, y_train)

    evaluator = Evaluator(trained_model)
    train_scores = evaluator.evaluate(X_train, y_train)
    test_scores = evaluator.evaluate(X_test, y_test)

    print(f"📊 Scores sur train :")
    Evaluator.print_scores(train_scores)
    print(f"📉 Scores sur test :")
    Evaluator.print_scores(test_scores)

    # Vérification overfitting
    diff = train_scores['R2'] - test_scores['R2']
    if diff > 0.2:
        print(f"⚠️ Overfitting détecté sur {name} (différence R2 = {diff:.3f})")
    else:
        print(f"✅ Pas d'overfitting détecté sur {name}")

    best_models.append((name, trained_model, test_scores['R2']))

# Étape 4 : Affichage des meilleurs modèles
best_models = sorted(best_models, key=lambda x: x[2], reverse=True)

print("\n🏆 Top 4 modèles par R2 test :")
for i, (name, model, score) in enumerate(best_models[:4], 1):
    print(f"{i}. {name} avec R2 = {score:.4f}")

# Sauvegarde des meilleurs modèles
for i, (name, model, score) in enumerate(best_models[:4], 1):
    joblib.dump(model, f"models/model_{i}_{name}.pkl")
    print(f"💾 Modèle {name} sauvegardé sous models/model_{i}_{name}.pkl")
