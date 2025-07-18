import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import Preprocessor
from src.modeling import ModelFactory, ModelTrainer
from src.evaluation import Evaluator
from src.database import save_to_database

def main():
    print("🚀 Début du pipeline")

    # -------------------- Prétraitement --------------------
    preprocessor = Preprocessor()
    df = preprocessor.load_data('data/appartements_data_db.csv')
    print("✅ Données chargées")

    df = preprocessor.clean_and_transform(df)
    print("✅ Données nettoyées et transformées")

    X, y = preprocessor.select_features(df)
    print(f"✅ Features sélectionnées : {X.shape}, Target : {y.shape}")

    # -------------------- Split train/test --------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print("✅ Données splittées")

    # -------------------- Initialisation des modèles --------------------
    models = ModelFactory().get_models()
    performances = []

    # -------------------- Entraînement et évaluation --------------------
    for name, model in models.items():
        print(f"\n🔧 Entraînement du modèle : {name}")
        trainer = ModelTrainer(model)
        trained_model = trainer.train(X_train, y_train)
        print(f"✅ Modèle {name} entraîné")

        evaluator = Evaluator(trained_model)
        scores = evaluator.evaluate(X_test, y_test)
        Evaluator.print_scores(scores)

        performances.append({
            'name': name,
            'model': trained_model,
            'R2': scores['R2']
        })

    # -------------------- Tri des modèles --------------------
    performances.sort(key=lambda x: x['R2'], reverse=True)
    print("\n🏆 Classement des modèles (par R²) :")
    for i, entry in enumerate(performances[:4]):
        print(f"{i+1}. {entry['name']} - R² = {entry['R2']:.4f}")

    # -------------------- Sauvegarde des 4 meilleurs modèles --------------------
    for i, entry in enumerate(performances[:4]):
        filename = f'models/best_model_{i+1}.pkl'
        joblib.dump(entry['model'], filename)
        print(f"💾 Modèle {entry['name']} sauvegardé sous {filename}")

    # -------------------- Sauvegarde des prédictions du meilleur modèle --------------------
    best_model = performances[0]['model']
    predictions = best_model.predict(X_test)
    df_preds = pd.DataFrame({'prediction': predictions})
    save_to_database(df_preds, 'predictions')
    print("📦 Prédictions du meilleur modèle sauvegardées dans la base")

    # -------------------- Sauvegarde des données nettoyées --------------------
    save_to_database(df, 'clean_data')
    print("📦 Données nettoyées sauvegardées dans la base")

    # -------------------- Sauvegarde du préprocesseur --------------------
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    print("🧠 Preprocessor sauvegardé dans models/preprocessor.pkl")

if __name__ == "__main__":
    main()
