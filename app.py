from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS


# Find the folder where this app.py file is located
BASE_DIR = Path(__file__).resolve().parent

# Location of our trained model
MODEL_PATH = BASE_DIR / "land_price_model.pkl"


# Create Flask application
app = Flask(__name__)

# Allow our HTML website to communicate with Python
CORS(app)


# Load the trained machine learning model
model = joblib.load(MODEL_PATH)


# Home route
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Land Price Prediction API is running!"
    })


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get data sent by the website
        input_data = request.get_json()

        # Create a table containing the user's input
        input_df = pd.DataFrame([{
            "region": input_data["region"],
            "land_type": input_data["land_type"],
            "road_access": input_data["road_access"],
            "utilities": input_data["utilities"],
            "area_sqft": float(input_data["area_sqft"]),
            "distance_to_city_km": float(input_data["distance_to_city_km"]),
            "infrastructure_score": float(input_data["infrastructure_score"]),
            "development_potential": float(input_data["development_potential"])
        }])

        # Ask the machine learning model to predict the price
        prediction = model.predict(input_df)[0]

        # Send prediction back to website
        return jsonify({
            "predicted_price": float(prediction)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


# Start the Flask server
if __name__ == "__main__":

    print("Land Price Prediction API is starting...")
    print("Open http://127.0.0.1:5000 in your browser")

    app.run(debug=True)