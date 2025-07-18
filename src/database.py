import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

def get_engine():
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    host = os.getenv('DB_HOST')
    port = os.getenv('DB_PORT')
    db = os.getenv('DB_NAME')
    
    url = f'postgresql://{user}:{password}@{host}:{port}/{db}'
    return create_engine(url)

engine = get_engine()

def save_to_database(df: pd.DataFrame, table_name: str, if_exists='replace'):
    try:
        df.to_sql(table_name, engine, index=False, if_exists=if_exists)
        print(f"Données enregistrées dans la table '{table_name}'")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde en base : {e}")
