from flask import Flask, render_template, request  # Import Flask components
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the saved model and scaler with error handling
try:
    model = joblib.load('xgboost_model.pkl')
except FileNotFoundError:
    raise FileNotFoundError("Model file 'xgboost_model.pkl' not found. Please ensure it is in the correct directory.")
except Exception as e:
    raise RuntimeError(f"An error occurred while loading the model: {e}")

try:
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    raise FileNotFoundError("Scaler file 'scaler.pkl' not found. Please ensure it is in the correct directory.")
except Exception as e:
    raise RuntimeError(f"An error occurred while loading the scaler: {e}")

# Define the features your model was trained on
features = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Year', 'Month', 'Week', 'Day_of_Week']
numerical_features_to_scale = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.json
        if not data:
            return {"error": "No input data provided"}, 400

        # Validate and convert input data
        input_data = []
        for feature in features:
            try:
                value = float(data[feature])  # Convert to float
                input_data.append(value)
            except ValueError:
                return {"error": f"Invalid value for {feature}. Must be a numeric value."}, 400

        # Convert to NumPy array
        input_data = np.array(input_data).reshape(1, -1)

        # Scale numerical features
        input_data[:, [features.index(f) for f in numerical_features_to_scale]] = scaler.transform(
            input_data[:, [features.index(f) for f in numerical_features_to_scale]]
        )

        # Make prediction
        prediction = model.predict(input_data)
        return {"prediction": prediction.tolist()}, 200

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}, 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))