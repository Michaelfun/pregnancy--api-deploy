import os
import joblib
import pandas as pd
from models import RuleBasedModel

def load_models():
    """Load all required models and return them in a dictionary."""
    # Load the main model
    MODEL_PATH = 'pregnancy_vitals_model_improved.joblib'
    print(f"Attempting to load model from: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = 'pregnancy_vitals_model.joblib'
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    model = joblib.load(MODEL_PATH)
    print(f"Successfully loaded model from: {MODEL_PATH}")
    
    # Load factor models
    factor_models = {}
    FACTOR_MODELS_INFO_PATH = 'factor_models_info.joblib'
    
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
                # Load the base model
                base_model = joblib.load(model_path)
                
                # Create a RuleBasedModel instance
                rule_based_model = RuleBasedModel(
                    ml_model=base_model,
                    factor=factor,
                    thresholds={'low': None, 'high': None},
                    normal_range=factor_ranges.get(factor, {})
                )
                
                factor_models[factor] = {
                    'model': rule_based_model,
                    'range': factor_ranges.get(factor, {})
                }
                print(f"Loaded {factor} model from {model_path}")
        
        print(f"Loaded {len(factor_models)} factor models")
    else:
        print("Factor models info not found. Individual factor risk assessment will not be available.")
    
    # Load feature columns
    if os.path.exists('feature_columns.txt'):
        with open('feature_columns.txt', 'r') as f:
            feature_columns = [line.strip() for line in f.readlines()]
    else:
        feature_columns = ['pulse', 'respiration', 'temperature', 'systolic', 'diastolic', 'oxygen']
    
    # Load normal ranges
    RANGES_PATH = 'normal_ranges.csv'
    if os.path.exists(RANGES_PATH):
        ranges_df = pd.read_csv(RANGES_PATH)
        normal_ranges = {}
        for _, row in ranges_df.iterrows():
            normal_ranges[row['vital_sign']] = {'min': row['min'], 'max': row['max']}
    else:
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
    
    return {
        'main_model': model,
        'factor_models': factor_models,
        'feature_columns': feature_columns,
        'normal_ranges': normal_ranges
    } 