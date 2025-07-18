# Simulateur Prix Immobilier ML

Ce projet permet d’estimer le prix des appartements au Maroc à partir de caractéristiques via un modèle de machine learning.

## Structure du projet

- `data/` : données immobilières brutes
- `models/` : modèles ML sauvegardés
- `notebooks/01_EDA.ipynb` : exploration des données
- `src/` : code source (prétraitement, modélisation, évaluation)
- `main.py` : script principal pour entraîner et sauvegarder le modèle
- `app.py` : API pour prédictions en ligne
- `Dockerfile` : conteneurisation
- `.github/workflows/main.yml` : pipeline CI/CD

## Installation

```bash
pip install -r requirements.txt
