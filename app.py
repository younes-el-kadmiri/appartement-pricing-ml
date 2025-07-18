from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()
model = joblib.load('models/best_model_1.pkl')

class Features(BaseModel):
    surface_area: float
    rooms: int
    location: str

@app.get("/")
def home():
    return {"message": "Simulateur prix ML"}

@app.post("/predict")
def predict(features: Features):
    df = pd.DataFrame([features.dict()])
    prediction = model.predict(df)
    return {"prix_estime": prediction[0]}
