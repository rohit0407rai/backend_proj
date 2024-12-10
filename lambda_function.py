import json
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

CORS(app)

def preprocess_input_for_prediction(input_data, zone='PowerConsumption_Zone1'):
    # Ensure the input data is in the correct format (as DataFrame)
    input_df = pd.DataFrame([input_data])

    # Set default values for missing columns (if not provided in input data)
    for col in ['PowerConsumption_Zone2', 'PowerConsumption_Zone3']:
        if col not in input_df.columns:
            input_df[col] = 0  # Assigning 0 as a default value

    # Calculate rolling mean and std (with min_periods=1 to avoid NaN issues)
    input_df[f'{zone}_rolling_mean'] = input_df[zone].rolling(window=5, min_periods=1).mean()
    input_df[f'{zone}_rolling_std'] = input_df[zone].rolling(window=5, min_periods=1).std()

    # Fill NaN values that might arise due to rolling mean/std calculation
    input_df.fillna(0, inplace=True)

    # Ensure the input columns match exactly with the model's training data
    expected_columns = [
        'Temperature', 'Humidity', 'WindSpeed',
        'GeneralDiffuseFlows', 'DiffuseFlows', 'PowerConsumption_Zone2',
        'PowerConsumption_Zone3', f'{zone}_rolling_mean', f'{zone}_rolling_std'
    ]

    # Reindex the input data to match the expected columns during training
    input_df = input_df[expected_columns]

    return input_df


# Function to check anomaly using the trained model
def check_anomaly_for_input(input_data, zone='PowerConsumption_Zone1', model_path='./models/power_consumption.pkl'):
    # Load the trained model, scaler, and PCA
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    pca = model_data['pca']

    # Preprocess the input data
    input_df = preprocess_input_for_prediction(input_data, zone)

    # Standardize the input data using the same scaler
    input_scaled = scaler.transform(input_df)

    # Apply PCA to the input data
    input_pca = pca.transform(input_scaled)

    # Use the Isolation Forest model to predict the anomaly
    anomaly = model.predict(input_pca)

    # Return whether the input is an anomaly or normal
    if anomaly == -1:
        return f"The input data for {zone} is an **anomaly**!"
    else:
        return f"The input data for {zone} is **normal**."

# Flask route to handle POST requests for anomaly checking
@app.route('/check-anomaly', methods=['POST'])
def check_anomaly():
    try:
        # Get input data from the request
        input_data = request.json

        # Call the check_anomaly_for_input function with the input data
        result = check_anomaly_for_input(input_data)

        # Return the result as a JSON response
        return jsonify({'message': result}), 200

    except Exception as e:
        # Catch any exceptions and return error message
        return jsonify({'message': f"Error occurred: {str(e)}"}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
