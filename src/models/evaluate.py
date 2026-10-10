# src/models/evaluate.py
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import pandas as pd

def evaluate_model_performance(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Computes comprehensive evaluation metrics for classification performance."""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'roc_auc': float(roc_auc_score(y_test, probs)),
        'f1_score': float(f1_score(y_test, preds)),
        'report': classification_report(y_test, preds, output_dict=True)
    }
    return metrics