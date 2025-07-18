# Base image
FROM python:3.10-slim

# Dossier de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet dans le conteneur
COPY . .

# Port exposé (si tu fais une API)
EXPOSE 8000

# Commande pour lancer le script principal ou serveur FastAPI/Flask
CMD ["python", "main.py"]
