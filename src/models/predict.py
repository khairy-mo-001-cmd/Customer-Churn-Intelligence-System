# src/models/predict.py
import pickle
import pandas as pd
from pathlib import Path

def load_model_bundle(model_path: Path):
    with open(model_path, 'rb') as f:
        artifact = pickle.load(f)
    return artifact

def run_batch_inference(df_features: pd.DataFrame, artifact) -> pd.DataFrame:
    binary_model = artifact.get('binary_model', artifact) if isinstance(artifact, dict) else artifact
    multi_model = artifact.get('multi_tier_model', artifact) if isinstance(artifact, dict) else artifact
    
    req_features = ['Frequency_Orders', 'Monetary_Spend', 'Total_Units', 
                    'Unique_Products', 'Avg_Order_Value', 'Avg_Units_Per_Order']
    
    X = df_features[req_features]
    results = df_features[['Customer_ID']].copy()
    
    results['Churn_Probability'] = binary_model.predict_proba(X)[:, 1].round(4)
    results['Is_Churned_Predicted'] = binary_model.predict(X)
    
    risk_map = {0: 'Active (<3 Mo)', 1: 'Moderate Risk (3-6 Mo)', 2: 'High Risk (6-9 Mo)', 3: 'Dormant (9+ Mo)'}
    raw_tiers = multi_model.predict(X)
    results['Risk_Horizon_Tier'] = [risk_map.get(t, 'Unknown') for t in raw_tiers]
    
    return results