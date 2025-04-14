from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import requests
import numpy as np
import pandas as pd
import stripe

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"], supports_credentials=True)

API_KEY = "b2275bbe9f6837aa1f94233731c588fc"
stripe.api_key = "sk_test_2l9yA6d83G8QKFRx8apsZ41j" 

df = pd.read_csv('crop_yield.csv')

# Load models
crop_model = joblib.load('crop_recommendation_model.pkl')
fert_model = joblib.load('fertilizer_prediction_model.pkl')
yield_model = joblib.load('crop_yield_model.pkl') 

# Load encoders
soil_encoder = joblib.load('soil_encoder.pkl')
crop_encoder = joblib.load('crop_encoder.pkl')
fertilizer_encoder = joblib.load('fertilizer_encoder.pkl')

# ---------------- CROP RECOMMENDATION API ----------------
@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    data = request.get_json()
    try:
        features = [
            data['N'], data['P'], data['K'],
            data['temperature'], data['humidity'],
            data['ph'], data['rainfall']
        ]
        prediction = crop_model.predict([features])
        return jsonify({'crop': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ---------------- FERTILIZER PREDICTION API ----------------
@app.route('/predict_fertilizer', methods=['POST'])
def predict_fertilizer():
    data = request.get_json()
    try:
        encoded_soil = soil_encoder.transform([data['soil_type']])[0]
        encoded_crop = crop_encoder.transform([data['crop_type']])[0]
        features = [
            data['temperature'], data['humidity'], data['moisture'],
            encoded_soil, encoded_crop,
            data['nitrogen'], data['phosphorus'], data['potassium']
        ]
        prediction = fert_model.predict([features])
        decoded_fertilizer = fertilizer_encoder.inverse_transform(prediction)
        return jsonify({'fertilizer': decoded_fertilizer[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ---------------- GET ALL CROP NAMES ----------------
@app.route('/get_crop_names', methods=['GET'])
def get_crop_names():
    try:
        crop_names = crop_encoder.classes_.tolist()
        return jsonify({'crop_names': crop_names})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---------------- GET ALL SOIL TYPES ----------------
@app.route('/get_soil_names', methods=['GET'])
def get_soil_names():
    try:
        soil_names = soil_encoder.classes_.tolist()
        return jsonify({'soil_names': soil_names})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---------------- RAINFALL & WEATHER FORECAST API ----------------
@app.route("/forecast", methods=["POST"])
def get_forecast():
    data = request.json
    location = data.get("location")

    if not location:
        return jsonify({"error": "Location is required"}), 400

    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?q={location}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        response.raise_for_status()

        weather_data = response.json()
        return jsonify({
            "city": weather_data["city"]["name"],
            "forecast": weather_data["list"]
        })
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500
        
# ---------------- YIELD PREDICTION API ----------------
@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    data = request.get_json()
    try:
        # Extract numerical inputs for the model
        annual_rainfall = float(data['Annual_Rainfall'])
        fertilizer = float(data['Fertilizer'])
        pesticide = float(data['Pesticide'])
        area = float(data['Area'])

        features = np.array([[annual_rainfall, fertilizer, pesticide, area]])
        prediction = yield_model.predict(features)

        return jsonify({'predicted_yield': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/dropdown-options', methods=['GET'])
def get_dropdown_options():
    try:
        options = {
            "Crop": sorted(df["Crop"].dropna().unique().tolist()) if "Crop" in df.columns else [],
            "Season": sorted(df["Season"].dropna().unique().tolist()) if "Season" in df.columns else [],
            "State": sorted(df["State"].dropna().unique().tolist()) if "State" in df.columns else [],
            "District": sorted(df["District"].dropna().unique().tolist()) if "District" in df.columns else []
        }

        return jsonify(options)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/create-checkout-session', methods=['POST', 'OPTIONS'])
def create_checkout_session():
    if request.method == 'OPTIONS':
        # Preflight request
        return '', 200

    data = request.get_json()
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'inr',
                    'product_data': {
                        'name': data['name'],
                    },
                    'unit_amount': data['amount'],
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url='http://localhost:3000/success',
            cancel_url='http://localhost:3000/cancel',
        )
        return jsonify({'url': session.url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
# ---------------- DEFAULT ----------------
@app.route('/')
def index():
    return "🌱 Crop, Fertilizer, and Yield Prediction API is running!"

if __name__ == '__main__':
    app.run(debug=True)
