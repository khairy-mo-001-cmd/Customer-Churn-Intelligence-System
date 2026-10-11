import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.features.build_rfm_features import FEATURE_COLUMNS
from src.features.build_training_set import build_labeled_set, temporal_cutoffs
from src.models.evaluate import evaluate_model_performance

logger = logging.getLogger(__name__)

HORIZONS = (90, 180, 270)  # "no purchase in the next 3 / 6 / 9 months"


def make_pipeline() -> Pipeline:
    """Gradient boosted trees. No scaler: trees are invariant to feature scaling."""
    return Pipeline([('classifier', HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=4, random_state=42))])


def _fit(X, y, calibrate: bool):
    if calibrate:
        return CalibratedClassifierCV(make_pipeline(), method='isotonic', cv=3).fit(X, y)
    return make_pipeline().fit(X, y)


def train_horizon_models(df: pd.DataFrame, horizons=HORIZONS, calibrate: bool = True):
    """Train one model per horizon with a strict temporal validation.

    For each horizon H:
      * train on customers at cutoff (snapshot - 2H), labels from the following H days;
      * evaluate on customers at cutoff (snapshot - H), labels from the final H days;
      * refit the final model on both labelled sets.
    Returns (bundle, metrics).
    """
    models, metrics = {}, {}
    for h in horizons:
        snapshot, train_cut, test_cut = temporal_cutoffs(df, h)
        train_set = build_labeled_set(df, train_cut, h)
        test_set = build_labeled_set(df, test_cut, h)
        Xtr, ytr = train_set[FEATURE_COLUMNS], train_set['Churn_Target']
        Xte, yte = test_set[FEATURE_COLUMNS], test_set['Churn_Target']

        raw_model = make_pipeline().fit(Xtr, ytr)
        raw_eval = evaluate_model_performance(raw_model, Xte, yte)
        use_calibration = False
        final_eval = raw_eval
        if calibrate:
            cal_model = _fit(Xtr, ytr, True)
            cal_eval = evaluate_model_performance(cal_model, Xte, yte)
            use_calibration = cal_eval['brier'] <= raw_eval['brier']
            final_eval = cal_eval if use_calibration else raw_eval

        metrics[h] = {
            'train_cutoff': str(train_cut), 'test_cutoff': str(test_cut),
            'n_train': int(len(train_set)), 'n_test': int(len(test_set)),
            'prevalence_test': float(yte.mean()),
            'roc_auc': final_eval['roc_auc'], 'pr_auc': final_eval['pr_auc'],
            'f1_score': final_eval['f1_score'], 'brier': final_eval['brier'],
            'brier_uncalibrated': raw_eval['brier'],
            'calibration_used': bool(use_calibration),
            'baseline_recency_roc_auc': float(roc_auc_score(yte, Xte['Recency_Days'])),
            'confusion_matrix': final_eval['confusion_matrix'],
        }
        logger.info('H=%sd  AUC=%.3f  PR-AUC=%.3f  Brier=%.3f  (recency baseline AUC=%.3f)',
                    h, metrics[h]['roc_auc'], metrics[h]['pr_auc'], metrics[h]['brier'],
                    metrics[h]['baseline_recency_roc_auc'])

        full = pd.concat([train_set, test_set], ignore_index=True)
        models[h] = _fit(full[FEATURE_COLUMNS], full['Churn_Target'], use_calibration)

    bundle = {
        'models': models,
        'horizons': list(horizons),
        'feature_columns': list(FEATURE_COLUMNS),
        'metrics': metrics,
        'snapshot_date': str(snapshot),
        'sklearn_version': sklearn.__version__,
        'trained_at': datetime.now(timezone.utc).isoformat(),
    }
    return bundle, metrics


def save_bundle(bundle: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    joblib.dump(bundle, tmp)
    tmp.replace(path)  # atomic replace: readers never see a half-written file
