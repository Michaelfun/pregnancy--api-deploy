"""
Pregnancy Vitals Monitoring - API Server
=======================================
This script implements a Flask API server that loads a pre-trained model
and exposes endpoints for making predictions.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import logging
import warnings
from models import RuleBasedModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Load the model and required data
MODEL_PATH = 'pregnancy_vitals_model_improved.joblib'
FACTOR_MODELS_INFO_PATH = 'factor_models_info.joblib'
RANGES_PATH = 'normal_ranges.csv'

print(f"Attempting to load model from: {MODEL_PATH}")

# Check if model exists
if not os.path.exists(MODEL_PATH):
    # Fall back to original model if improved model doesn't exist
    print(f"Improved model not found, falling back to original model")
    MODEL_PATH = 'pregnancy_vitals_model.joblib'
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Please run main.py or model_training.py first.")

# Load the model
model = joblib.load(MODEL_PATH)
print(f"Successfully loaded model from: {MODEL_PATH}")

# Load factor models if available
factor_models = {}
if os.path.exists(FACTOR_MODELS_INFO_PATH):
    factor_models_info = joblib.load(FACTOR_MODELS_INFO_PATH)
    print(f"Loading factor models from: {factor_models_info['model_paths']}")
    
    # Load normal ranges
    if os.path.exists(factor_models_info['ranges_path']):
        factor_ranges = joblib.load(factor_models_info['ranges_path'])
    else:
        factor_ranges = {}
    
    # Load each factor model
    for factor, model_path in factor_models_info['model_paths'].items():
        if os.path.exists(model_path):
            factor_models[factor] = {
                'model': joblib.load(model_path),
                'range': factor_ranges.get(factor, {})
            }
            print(f"Loaded {factor} model from {model_path}")
    
    print(f"Loaded {len(factor_models)} factor models")
else:
    print("Factor models info not found. Individual factor risk assessment will not be available.")

# Load the feature columns
if os.path.exists('feature_columns.txt'):
    with open('feature_columns.txt', 'r') as f:
        feature_columns = [line.strip() for line in f.readlines()]
else:
    # Default feature columns if file doesn't exist
    feature_columns = ['pulse', 'respiration', 'temperature', 'systolic', 'diastolic', 'oxygen']

# Load the normal ranges
if os.path.exists(RANGES_PATH):
    ranges_df = pd.read_csv(RANGES_PATH)
    normal_ranges = {}
    for _, row in ranges_df.iterrows():
        normal_ranges[row['vital_sign']] = {'min': row['min'], 'max': row['max']}
else:
    # Default ranges if file doesn't exist
    normal_ranges = {
        'pulse': {'min': 60, 'max': 100},
        'respiration': {'min': 12, 'max': 20},
        'temperature': {'min': 35, 'max': 38},
        'systolic': {'min': 80, 'max': 120},
        'diastolic': {'min': 80, 'max': 120},
        'oxygen': {'min': 95, 'max': 100},
        'glucose': {'min': 3, 'max': 200},
        'height': {'min': 120, 'max': 200},
        'weight': {'min': 30, 'max': 200}
    }

print("Model and data loaded successfully")
print(f"Feature columns: {feature_columns}")
print(f"Normal ranges: {normal_ranges}")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making predictions on a single record
    """
    try:
        # Get data from request
        data = request.json
        
        # Create a DataFrame with the input data
        input_df = pd.DataFrame([data])
        
        # Ensure all required features are present
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = np.nan
        
        # Make prediction
        risk_level = model.predict(input_df[feature_columns])[0]
        
        # Get probability scores
        probabilities = model.predict_proba(input_df[feature_columns])[0]
        prob_dict = {cls: float(prob) for cls, prob in zip(model.named_steps['classifier'].classes_, probabilities)}
        
        # Add analysis of each vital sign
        analysis = {}
        for col in feature_columns:
            if col in data and data[col] is not None:
                if pd.notna(data[col]) and data[col] != 0:
                    # Determine if value is within normal range
                    status = "normal"
                    min_val = normal_ranges[col]['min']
                    max_val = normal_ranges[col]['max']
                    
                    if min_val is not None and data[col] < min_val:
                        status = "below normal"
                    elif max_val is not None and data[col] > max_val:
                        status = "above normal"
                    
                    # Create analysis object for this vital sign
                    vital_analysis = {
                        "value": data[col],
                        "status": status,
                        "normal_range": f"{min_val}-{max_val}"
                    }
                    
                    # Always use factor models if available, directly from the model
                    if col in factor_models:
                        factor_model = factor_models[col]['model']
                        
                        # Create a single feature DataFrame
                        X_factor = pd.DataFrame([{col: data[col]}])
                        
                        # Get prediction from the model
                        factor_risk = factor_model.predict(X_factor)[0]
                        
                        # Get probability scores from the model
                        factor_probs = factor_model.predict_proba(X_factor)[0]
                        factor_prob_dict = {cls: float(prob) for cls, prob in zip(factor_model.classes_, factor_probs)}
                        
                        # Add model-based assessment to the response
                        vital_analysis["factor_risk"] = factor_risk
                        vital_analysis["factor_risk_probabilities"] = factor_prob_dict
                    
                    analysis[col] = vital_analysis
        
        # Prepare response
        response = {
            "risk_level": risk_level,
            "risk_probabilities": prob_dict,
            "vital_signs_analysis": analysis
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Endpoint for making predictions on multiple records at once
    """
    try:
        # Get batch data from request
        batch_data = request.json
        
        # Create a DataFrame with the input data
        batch_df = pd.DataFrame(batch_data)
        
        # Ensure all required features are present
        for col in feature_columns:
            if col not in batch_df.columns:
                batch_df[col] = np.nan
        
        # Make predictions
        risk_levels = model.predict(batch_df[feature_columns]).tolist()
        
        # Get detailed predictions with analysis
        detailed_results = []
        for i, record in batch_df.iterrows():
            # Convert to dictionary and handle numpy types
            record_dict = record.to_dict()
            for k, v in record_dict.items():
                if isinstance(v, (np.int64, np.float64)):
                    record_dict[k] = float(v)
            
            # Add analysis of each vital sign
            analysis = {}
            for col in feature_columns:
                if col in record_dict and record_dict[col] is not None:
                    if pd.notna(record_dict[col]) and record_dict[col] != 0:
                        # Determine if value is within normal range
                        status = "normal"
                        min_val = normal_ranges[col]['min']
                        max_val = normal_ranges[col]['max']
                        
                        if min_val is not None and record_dict[col] < min_val:
                            status = "below normal"
                        elif max_val is not None and record_dict[col] > max_val:
                            status = "above normal"
                        
                        # Create analysis object for this vital sign
                        vital_analysis = {
                            "value": record_dict[col],
                            "status": status,
                            "normal_range": f"{min_val}-{max_val}"
                        }
                        
                        # Always use factor models if available, directly from the model
                        if col in factor_models:
                            factor_model = factor_models[col]['model']
                            
                            # Create a single feature DataFrame
                            X_factor = pd.DataFrame([{col: record_dict[col]}])
                            
                            # Get prediction from the model
                            factor_risk = factor_model.predict(X_factor)[0]
                            
                            # Get probability scores from the model
                            factor_probs = factor_model.predict_proba(X_factor)[0]
                            factor_prob_dict = {cls: float(prob) for cls, prob in zip(factor_model.classes_, factor_probs)}
                            
                            # Add model-based assessment to the response
                            vital_analysis["factor_risk"] = factor_risk
                            vital_analysis["factor_risk_probabilities"] = factor_prob_dict
                        
                        analysis[col] = vital_analysis
            
            # Add to detailed results
            detailed_results.append({
                "record": record_dict,
                "risk_level": risk_levels[i],
                "vital_signs_analysis": analysis
            })
        
        # Prepare response
        response = {
            "risk_levels": risk_levels,
            "detailed_results": detailed_results,
            "count": len(risk_levels)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify API is running
    """
    return jsonify({
        "status": "healthy", 
        "model": "pregnancy_vitals_model",
        "features": feature_columns
    })

@app.route('/ranges', methods=['GET'])
def get_ranges():
    """
    Endpoint to retrieve the normal ranges for vital signs
    """
    return jsonify({
        "normal_ranges": normal_ranges
    })

@app.route('/', methods=['GET'])
def home():
    """
    API home page
    """
    return jsonify({
        "api": "Pregnancy Vitals Monitoring API",
        "version": "1.0",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "API information"},
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/ranges", "method": "GET", "description": "Get normal ranges for vital signs"},
            {"path": "/predict", "method": "POST", "description": "Predict risk level for a single record"},
            {"path": "/batch_predict", "method": "POST", "description": "Predict risk levels for multiple records"}
        ]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)