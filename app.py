from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Esto le da permiso a tu Frontend de conectarse
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

modelo = joblib.load('model/commodity-v1.joblib')

class CommodityData(BaseModel):
    natural_gas: float
    wti_crude: float
    copper: float
    silver: float

@app.get("/")
def inicio():
    return {"mensaje": "API de Commodities lista"}

@app.post("/predict")
def predecir(data: CommodityData):
    df = pd.DataFrame([[data.natural_gas, data.wti_crude, data.copper, data.silver]],
                      columns=['NATURAL GAS', 'WTI CRUDE', 'COPPER', 'SILVER'])
    prediccion = modelo.predict(df)
    return {"prediccion_gold": float(prediccion[0])}