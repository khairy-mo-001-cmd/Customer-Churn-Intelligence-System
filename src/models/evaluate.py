from sklearn.metrics import (average_precision_score, brier_score_loss, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)


def evaluate_model_performance(model, X_test, y_test) -> dict:
    """Binary-classification metrics. Raises if y_test contains a single class."""
    if len(set(y_test)) < 2:
        raise ValueError('y_test must contain both classes to compute ROC-AUC')
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return {
        'roc_auc': float(roc_auc_score(y_test, probs)),
        'pr_auc': float(average_precision_score(y_test, probs)),
        'f1_score': float(f1_score(y_test, preds, zero_division=0)),
        'brier': float(brier_score_loss(y_test, probs)),
        'confusion_matrix': confusion_matrix(y_test, preds).tolist(),
        'report': classification_report(y_test, preds, output_dict=True, zero_division=0),
    }
