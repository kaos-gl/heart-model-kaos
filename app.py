from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
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