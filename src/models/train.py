# src/models/train.py
import pickle
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.utils.config import MODELS_DIR

def train_champion_pipeline(X_train: pd.DataFrame, y_train: pd.Series, multi_tier_y: pd.Series = None):
    """Trains the champion binary churn classifier and multi-horizon risk classifier."""
    binary_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, random_state=42))
    ])
    
    binary_pipeline.fit(X_train, y_train)
    
    multi_tier_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, random_state=42))
    ])
    
    if multi_tier_y is not None:
        multi_tier_pipeline.fit(X_train, multi_tier_y)
    else:
        multi_tier_pipeline.fit(X_train, y_train)
        
    print("[OK] Champion models trained successfully.")
    return binary_pipeline, multi_tier_pipeline