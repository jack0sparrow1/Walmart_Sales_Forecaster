from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')

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
        
        # Create a DataFrame with the received data
        input_df = pd.DataFrame([data])
        
        # Ensure the columns are in the correct order
        input_df = input_df[features]

        # Apply the same scaling as the training data
        input_df[numerical_features_to_scale] = scaler.transform(input_df[numerical_features_to_scale])

        # Make the prediction
        prediction = model.predict(input_df)

        # Return the prediction as a JSON response
        return jsonify({'prediction': np.round(prediction[0], 2)})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # You might want to change debug=False in a production environment
    app.run(debug=True)