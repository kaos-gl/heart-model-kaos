import pandas as pd
from xgboost import XGBRegressor
from joblib import dump

# Leemos el dataset descargado genéricamente
df = pd.read_csv('data/dataset.csv')

# Usaremos estas 4 variables para predecir el GOLD
features = ['NATURAL GAS', 'WTI CRUDE', 'COPPER', 'SILVER']
df = df.dropna(subset=features + ['GOLD'])

X = df[features]
y = df['GOLD']

print('Entrenando modelo XGBoost...')
model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
model.fit(X, y)

dump(model, 'model/commodity-v1.joblib')
print('Modelo guardado.')