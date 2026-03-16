# rule_based_model.py
import numpy as np

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
