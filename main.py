"""
Pregnancy Vitals Monitoring - Model Training and API Endpoint
======================================================
This script performs the following tasks:
1. Data loading and preprocessing
2. Model training
3. Model evaluation
4. API endpoint creation

Requirements:
- pandas
- numpy
- scikit-learn
- flask
- openpyxl
- joblib
"""

import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from flask import Flask, request, jsonify

# ===========================
# DATA LOADING AND PREPROCESSING
# ===========================

def load_and_preprocess_data(pregnancy_vitals_path, signal_ranges_path):
    """
    Load and preprocess the pregnancy vitals data using the normal ranges
    """
    print(f"Loading data from {pregnancy_vitals_path}...")
    
    # Load pregnancy vitals data
    vitals_df = pd.read_excel(pregnancy_vitals_path)
    print(f"Loaded {len(vitals_df)} records")

    # Load signal ranges data
    ranges_df = pd.read_excel(signal_ranges_path, header=2)  # Skip the first two rows
    
    # Create a dictionary of normal ranges
    normal_ranges = {}
    for _, row in ranges_df.iterrows():
        if pd.notna(row['SIGN']):
            sign = row['SIGN'].lower()
            
            # Map the sign names to column names
            if 'temperature' in sign:
                col = 'temperature'
            elif 'pulse' in sign:
                col = 'pulse'
            elif 'respiration' in sign or 'mapafu' in sign:
                col = 'respiration'
            elif 'sytotic' in sign or 'high pressure' in sign:
                col = 'systolic'
            elif 'diastotic' in sign or 'low pressure' in sign:
                col = 'diastolic'
            elif 'oxygen' in sign:
                col = 'oxygen'
            else:
                continue
                
            # Store min and max values (handling NaN)
            min_val = row['MIN_VALUE'] if pd.notna(row['MIN_VALUE']) else None
            max_val = row['MAX_VALUE'] if pd.notna(row['MAX_VALUE']) else None
            
            normal_ranges[col] = {'min': min_val, 'max': max_val}
    
    # Fill in missing ranges with reasonable values
    fallback_ranges = {
        'pulse': {'min': 40, 'max': 140},
        'respiration': {'min': 8, 'max': 30},
        'temperature': {'min': 35, 'max': 40},
        'systolic': {'min': 80, 'max': 180},
        'diastolic': {'min': 40, 'max': 120},
        'glucose': {'min': 3, 'max': 200},
        'oxygen': {'min': 80, 'max': 100},
        'height': {'min': 120, 'max': 200},
        'weight': {'min': 30, 'max': 200}
    }
    
    for col, ranges in fallback_ranges.items():
        if col not in normal_ranges:
            normal_ranges[col] = ranges
        else:
            # Replace None values with fallbacks
            if normal_ranges[col]['min'] is None:
                normal_ranges[col]['min'] = fallback_ranges[col]['min']
            if normal_ranges[col]['max'] is None:
                normal_ranges[col]['max'] = fallback_ranges[col]['max']

    print("Normal ranges:", normal_ranges)
    
    # Clean the data
    print("Cleaning data...")
    
    # 1. Handle date formatting
    try:
        # Try multiple date formats
        vitals_df['create_date'] = pd.to_datetime(vitals_df['create_date'], errors='coerce')
    except Exception as e:
        print(f"Warning: Error converting dates - {e}")
    
    # 2. Filter out implausible values and replace with NaN
    for col, ranges in normal_ranges.items():
        if col in vitals_df.columns:
            # Replace zeros with NaN (except for glucose which can be zero)
            if col != 'glucose':
                vitals_df[col] = vitals_df[col].replace(0, np.nan)
            
            # Replace values outside of range with NaN
            if ranges['min'] is not None:
                vitals_df.loc[vitals_df[col] < ranges['min'], col] = np.nan
            if ranges['max'] is not None:
                vitals_df.loc[vitals_df[col] > ranges['max'], col] = np.nan
    
    # 3. Create target variable based on vital signs being within normal ranges
    print("Creating classification labels...")
    
    # Initialize risk_level column 
    vitals_df['risk_level'] = 'normal'
    
    # Function to determine risk level based on vital signs
    def determine_risk_level(row):
        high_risk_count = 0
        medium_risk_count = 0
        vital_columns = ['pulse', 'respiration', 'temperature', 'systolic', 'diastolic', 'oxygen']
        
        for col in vital_columns:
            if pd.isna(row[col]):
                continue
                
            # Get range for this vital sign
            if col in normal_ranges:
                min_val = normal_ranges[col]['min']
                max_val = normal_ranges[col]['max']
                
                # Check if value is outside range
                if min_val is not None and row[col] < min_val:
                    if col in ['temperature', 'oxygen', 'systolic', 'diastolic']:
                        high_risk_count += 1  # These are more critical
                    else:
                        medium_risk_count += 1
                        
                if max_val is not None and row[col] > max_val:
                    if col in ['temperature', 'oxygen', 'systolic', 'diastolic']:
                        high_risk_count += 1
                    else:
                        medium_risk_count += 1
        
        # Determine overall risk level
        if high_risk_count > 0:
            return 'high'
        elif medium_risk_count > 1:  # More than one medium risk factor
            return 'medium'
        elif medium_risk_count > 0:  # One medium risk factor
            return 'low'
        else:
            return 'normal'
    
    # Apply the risk level function to each row
    vitals_df['risk_level'] = vitals_df.apply(determine_risk_level, axis=1)
    
    # Print distribution of risk levels
    risk_counts = vitals_df['risk_level'].value_counts()
    print("Risk level distribution:")
    print(risk_counts)
    
    return vitals_df, normal_ranges

# ===========================
# MODEL TRAINING
# ===========================

def train_model(vitals_df):
    """
    Train a model to predict risk levels based on vital signs
    """
    print("\nPreparing model training data...")
    
    # Select features and target
    feature_columns = ['pulse', 'respiration', 'temperature', 'systolic', 'diastolic', 'oxygen']
    X = vitals_df[feature_columns]
    y = vitals_df['risk_level']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on the test set
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model
    model_filename = 'pregnancy_vitals_model.joblib'
    joblib.dump(pipeline, model_filename)
    print(f"Model saved to {model_filename}")
    
    return pipeline, feature_columns

# ===========================
# API ENDPOINT
# ===========================

def create_api(model, feature_columns, normal_ranges):
    """
    Create a Flask API to serve predictions
    """
    app = Flask(__name__)
    
    @app.route('/predict', methods=['POST'])
    def predict():
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
            prob_dict = {cls: prob for cls, prob in zip(model.classes_, probabilities)}
            
            # Add analysis of each vital sign
            analysis = {}
            for col in feature_columns:
                if col in data and data[col] is not None:
                    if pd.notna(data[col]) and data[col] != 0:
                        status = "normal"
                        min_val = normal_ranges[col]['min']
                        max_val = normal_ranges[col]['max']
                        
                        if min_val is not None and data[col] < min_val:
                            status = "below normal"
                        elif max_val is not None and data[col] > max_val:
                            status = "above normal"
                            
                        analysis[col] = {
                            "value": data[col],
                            "status": status,
                            "normal_range": f"{min_val}-{max_val}"
                        }
            
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
            
            # Prepare response
            response = {
                "risk_levels": risk_levels,
                "count": len(risk_levels)
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "healthy", "model": "pregnancy_vitals_model"})
    
    return app

# ===========================
# MAIN EXECUTION
# ===========================

def main():
    """
    Main execution function
    """
    try:
        # Paths to data files
        pregnancy_vitals_path = 'pregnancy_vitals.xlsx'
        signal_ranges_path = 'Signal for Maternal monitoring.xlsx'
        
        # Step 1: Load and preprocess data
        vitals_df, normal_ranges = load_and_preprocess_data(pregnancy_vitals_path, signal_ranges_path)
        
        # Step 2: Train model
        model, feature_columns = train_model(vitals_df)
        
        # Step 3: Create API
        app = create_api(model, feature_columns, normal_ranges)
        
        # Step 4: Run the API (in production, use a proper WSGI server)
        print("\nStarting API server...")
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()