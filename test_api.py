import requests
import pandas as pd

df = pd.read_csv("../energy-ops/data/01_raw/electricityConsumptionAndProductioction.csv")  # Ajustez le chemin si nécessaire
df["DateTime"] = pd.to_datetime(df["DateTime"])
df["hour"] = df["DateTime"].dt.hour
df["dayofweek"] = df["DateTime"].dt.dayofweek
df = df.drop(columns=["DateTime"])
data = df.tail(24).values.tolist()
response = requests.post("http://127.0.0.1:5000/predict", json=data)
print(response.json())