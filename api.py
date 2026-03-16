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
import threading
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import logging
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class RuleBasedModel:
    def __init__(self, ml_model, factor, thresholds, normal_range):
        self.ml_model = ml_model
        self.factor = factor
        self.thresholds = thresholds
        self.normal_range = normal_range
        self.classes_ = ml_model.classes_

    def predict(self, X):
        value = X.iloc[0][self.factor]
        if self.thresholds['low'] is not None and value < self.thresholds['low']:
            return np.array(['high'])
        elif self.thresholds['high'] is not None and value > self.thresholds['high']:
            return np.array(['high'])
        else:
            return self.ml_model.predict(X)

    def predict_proba(self, X):
        value = X.iloc[0][self.factor]
        if (self.thresholds['low'] is not None and value < self.thresholds['low']) or \
           (self.thresholds['high'] is not None and value > self.thresholds['high']):
            if self.classes_[0] == 'normal':
                return np.array([[0.0, 1.0]])
            else:
                return np.array([[1.0, 0.0]])
        else:
            min_val = self.normal_range['min']
            max_val = self.normal_range['max']
            if min_val <= value <= max_val:
                if self.classes_[0] == 'normal':
                    return np.array([[0.99, 0.01]])
                else:
                    return np.array([[0.01, 0.99]])
            else:
                return self.ml_model.predict_proba(X)


app = Flask(__name__)
CORS(app)

# ── Global model state ────────────────────────────────────────────────────────
model = None
CLASSIFIER_CLASSES = None
factor_models = {}
feature_columns = []
normal_ranges = {}
_models_ready = False
_models_error = None
_models_lock = threading.Lock()
# ─────────────────────────────────────────────────────────────────────────────


def _load_models():
    global model, CLASSIFIER_CLASSES, factor_models, feature_columns, normal_ranges
    global _models_ready, _models_error

    try:
        MODEL_PATH = 'pregnancy_vitals_model_improved.joblib'
        FACTOR_MODELS_INFO_PATH = 'factor_models_info.joblib'
        RANGES_PATH = 'normal_ranges.csv'

        logger.info(f"Attempting to load model from: {MODEL_PATH}")

        if not os.path.exists(MODEL_PATH):
            logger.info("Improved model not found, falling back to original model")
            MODEL_PATH = 'pregnancy_vitals_model.joblib'
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH}.")

        model = joblib.load(MODEL_PATH)
        logger.info(f"Successfully loaded model from: {MODEL_PATH}")

        try:
            CLASSIFIER_CLASSES = list(model.named_steps['classifier'].classes_)
        except Exception:
            CLASSIFIER_CLASSES = None

        _factor_models = {}
        if os.path.exists(FACTOR_MODELS_INFO_PATH):
            factor_models_info = joblib.load(FACTOR_MODELS_INFO_PATH)
            logger.info(f"Loading factor models from: {factor_models_info['model_paths']}")

            factor_ranges = {}
            if os.path.exists(factor_models_info['ranges_path']):
                factor_ranges = joblib.load(factor_models_info['ranges_path'])

            for factor, model_path in factor_models_info['model_paths'].items():
                if os.path.exists(model_path):
                    _factor_models[factor] = {
                        'model': joblib.load(model_path),
                        'range': factor_ranges.get(factor, {})
                    }
                    logger.info(f"Loaded {factor} model from {model_path}")
        else:
            logger.warning("Factor models info not found.")

        factor_models = _factor_models

        if os.path.exists('feature_columns.txt'):
            with open('feature_columns.txt', 'r') as f:
                feature_columns = [line.strip() for line in f.readlines()]
        else:
            feature_columns = ['pulse', 'respiration', 'temperature', 'systolic', 'diastolic', 'oxygen']

        if os.path.exists(RANGES_PATH):
            ranges_df = pd.read_csv(RANGES_PATH)
            for _, row in ranges_df.iterrows():
                normal_ranges[row['vital_sign']] = {'min': row['min'], 'max': row['max']}
        else:
            normal_ranges.update({
                'pulse':       {'min': 60,  'max': 100},
                'respiration': {'min': 12,  'max': 20},
                'temperature': {'min': 35,  'max': 38},
                'systolic':    {'min': 80,  'max': 120},
                'diastolic':   {'min': 80,  'max': 120},
                'oxygen':      {'min': 95,  'max': 100},
                'glucose':     {'min': 3,   'max': 200},
                'height':      {'min': 120, 'max': 200},
                'weight':      {'min': 30,  'max': 200},
            })

        logger.info("All models loaded successfully")

        with _models_lock:
            _models_ready = True

    except Exception as exc:
        logger.error(f"Failed to load models: {exc}", exc_info=True)
        with _models_lock:
            _models_error = str(exc)


# Start background model loading before first request
threading.Thread(target=_load_models, daemon=True).start()


def _parse_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    return default


def _require_models():
    with _models_lock:
        if _models_error:
            return jsonify({"error": f"Model failed to load: {_models_error}"}), 503
        if not _models_ready:
            return jsonify({"error": "Models are still loading, please retry in a few seconds."}), 503
    return None


@app.route('/predict', methods=['POST'])
def predict():
    early = _require_models()
    if early:
        return early
    try:
        data = request.json
        details = _parse_bool(request.args.get("details"), default=True)
        include_factor_models = _parse_bool(request.args.get("factor_models"), default=True)

        input_df = pd.DataFrame(
            [[data.get(col, np.nan) for col in feature_columns]],
            columns=feature_columns
        )

        risk_level = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        classes = CLASSIFIER_CLASSES or list(getattr(model, "classes_", []))
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}

        response = {"risk_level": risk_level, "risk_probabilities": prob_dict}

        if details:
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
                        vital_analysis = {
                            "value": data[col],
                            "status": status,
                            "normal_range": f"{min_val}-{max_val}"
                        }
                        if include_factor_models and col in factor_models:
                            factor_model = factor_models[col]['model']
                            X_factor = pd.DataFrame([{col: data[col]}])
                            factor_risk = factor_model.predict(X_factor)[0]
                            factor_probs = factor_model.predict_proba(X_factor)[0]
                            vital_analysis["factor_risk"] = factor_risk
                            vital_analysis["factor_risk_probabilities"] = {
                                cls: float(prob)
                                for cls, prob in zip(factor_model.classes_, factor_probs)
                            }
                        analysis[col] = vital_analysis
            response["vital_signs_analysis"] = analysis

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    early = _require_models()
    if early:
        return early
    try:
        batch_data = request.json
        batch_df = pd.DataFrame(batch_data)
        for col in feature_columns:
            if col not in batch_df.columns:
                batch_df[col] = np.nan

        risk_levels = model.predict(batch_df[feature_columns]).tolist()
        detailed_results = []

        for i, record in batch_df.iterrows():
            record_dict = record.to_dict()
            for k, v in record_dict.items():
                if isinstance(v, (np.int64, np.float64)):
                    record_dict[k] = float(v)

            analysis = {}
            for col in feature_columns:
                if col in record_dict and record_dict[col] is not None:
                    if pd.notna(record_dict[col]) and record_dict[col] != 0:
                        status = "normal"
                        min_val = normal_ranges[col]['min']
                        max_val = normal_ranges[col]['max']
                        if min_val is not None and record_dict[col] < min_val:
                            status = "below normal"
                        elif max_val is not None and record_dict[col] > max_val:
                            status = "above normal"
                        vital_analysis = {
                            "value": record_dict[col],
                            "status": status,
                            "normal_range": f"{min_val}-{max_val}"
                        }
                        if col in factor_models:
                            factor_model = factor_models[col]['model']
                            X_factor = pd.DataFrame([{col: record_dict[col]}])
                            factor_risk = factor_model.predict(X_factor)[0]
                            factor_probs = factor_model.predict_proba(X_factor)[0]
                            vital_analysis["factor_risk"] = factor_risk
                            vital_analysis["factor_risk_probabilities"] = {
                                cls: float(prob)
                                for cls, prob in zip(factor_model.classes_, factor_probs)
                            }
                        analysis[col] = vital_analysis

            detailed_results.append({
                "record": record_dict,
                "risk_level": risk_levels[i],
                "vital_signs_analysis": analysis
            })

        return jsonify({
            "risk_levels": risk_levels,
            "detailed_results": detailed_results,
            "count": len(risk_levels)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/health', methods=['GET'])
def health_check():
    """
    Returns 200 immediately — even while models are still loading.
    This keeps Render's port-scan happy right away.
    """
    with _models_lock:
        ready = _models_ready
        error = _models_error

    if error:
        return jsonify({"status": "error", "detail": error}), 500
    if not ready:
        return jsonify({
            "status": "loading",
            "detail": "Models are still loading, predictions not yet available."
        }), 200   # 200 so Render sees the port as open

    return jsonify({
        "status": "healthy",
        "model": "pregnancy_vitals_model",
        "features": feature_columns
    })


@app.route('/ranges', methods=['GET'])
def get_ranges():
    early = _require_models()
    if early:
        return early
    return jsonify({"normal_ranges": normal_ranges})


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "api": "Pregnancy Vitals Monitoring API",
        "version": "1.0",
        "endpoints": [
            {"path": "/",              "method": "GET",  "description": "API information"},
            {"path": "/health",        "method": "GET",  "description": "Health check"},
            {"path": "/ranges",        "method": "GET",  "description": "Get normal ranges"},
            {"path": "/predict",       "method": "POST", "description": "Single-record prediction"},
            {"path": "/batch_predict", "method": "POST", "description": "Batch prediction"},
        ]
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
