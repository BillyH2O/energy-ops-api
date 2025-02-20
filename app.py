# energy_forecast_api/app.py
from flask import Flask, request, jsonify
from src.model import EnergyModel

app = Flask(__name__)

# Charger le modèle à l'initialisation
model = EnergyModel()

@app.route('/', methods=['GET'])
def home():
    return "Energy Forecast API !", 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json()
        if not isinstance(body, list):
            return jsonify({"error": "Input must be a list of observations"}), 400
        
        # Supposons que seq_length est fixé à 24 (ajustez selon votre modèle)
        prediction = model.predict(body, seq_length=24)
        return jsonify({"prediction": float(prediction)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)