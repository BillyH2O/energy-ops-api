# energy_forecast_api/src/model.py
import os
import mlflow
import joblib
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
import tensorflow  # Ajout explicite
from tensorflow.keras.models import load_model

class EnergyModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()

    def load_model(self):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_SERVER"))
        client = MlflowClient()
        model_version = client.get_latest_versions(os.getenv("MLFLOW_REGISTRY_NAME"), [os.getenv("ENV")])[0]
        run_id = model_version.run_id

        # Charger le modèle Keras
        self.model = mlflow.keras.load_model(f"runs:/{run_id}/model")
        
        # Charger le scaler
        scaler_path = client.download_artifacts(run_id, "transformations/scaler.pkl")
        self.scaler = joblib.load(scaler_path)

    def preprocess(self, data: list, seq_length: int) -> np.ndarray:
        # Convertir la liste JSON en DataFrame
        columns = ["Consumption", "Production", "Nuclear", "Wind", "Hydroelectric",
                   "Oil and Gas", "Coal", "Solar", "Biomass", "hour", "dayofweek"]
        df = pd.DataFrame(data, columns=columns)
        
        # Ajouter les features temporelles comme dans le pipeline
        df["hour"] = df["hour"].astype(int)
        df["dayofweek"] = df["dayofweek"].astype(int)
        
        # Appliquer le MinMaxScaler
        df_scaled = pd.DataFrame(self.scaler.transform(df), columns=df.columns)
        
        # Créer une séquence
        if len(df_scaled) >= seq_length:
            X = np.array([df_scaled.iloc[-seq_length:].values])
            return X
        else:
            raise ValueError(f"Data length ({len(df_scaled)}) must be >= seq_length ({seq_length})")

    def predict(self, data: list, seq_length: int = 24) -> np.ndarray:
        if not self.model or not self.scaler:
            raise Exception("Model or scaler not loaded.")
        
        X = self.preprocess(data, seq_length)
        pred_scaled = self.model.predict(X)
        
        # Inverser la mise à l'échelle pour la colonne Consumption uniquement
        dummy_array = np.zeros((1, len(self.scaler.feature_names_in_)))
        dummy_array[0, 0] = pred_scaled[0, 0]  # Consumption est en index 0
        pred_unscaled = self.scaler.inverse_transform(dummy_array)[0, 0]
        
        return pred_unscaled