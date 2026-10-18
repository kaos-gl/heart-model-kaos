from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Crear la aplicación que Render y Uvicorn están buscando
app = FastAPI()

# 2. Cargar el modelo que fue generado por train.py
modelo = joblib.load('model/heart-disease-v1.joblib')

# 3. Definir la estructura exacta de datos basándonos en heart-disease.csv
class Paciente(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.get("/")
def inicio():
    return {"mensaje": "API del modelo Heart Disease activa"}

@app.post("/predict")
def predecir(paciente: Paciente):
    # Convertir los datos recibidos a un DataFrame de Pandas
    datos = pd.DataFrame([paciente.dict()])
    # Realizar la predicción con el modelo
    prediccion = modelo.predict(datos)
    return {"prediccion_target": int(prediccion[0])}