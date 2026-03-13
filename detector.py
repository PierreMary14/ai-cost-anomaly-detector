import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Dict

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CostAnomalyDetector:
    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        
    def train_and_detect(self, data: List[Dict]) -> pd.DataFrame:
        df = pd.DataFrame(data)
        logging.info("Training Isolation Forest model on billing data...")
        
        # Fit model
        df['anomaly_score'] = self.model.fit_predict(df[['spend']])
        
        # -1 is anomaly, 1 is normal
        df['is_anomaly'] = df['anomaly_score'].apply(lambda x: True if x == -1 else False)
        return df

if __name__ == "__main__":
    # Sample Cloud Spend Data
    raw_data = [{'day': i, 'spend': 100 + np.random.randint(-10, 10)} for i in range(30)]
    raw_data.append({'day': 31, 'spend': 1500})  # Critical Anomaly

    detector = CostAnomalyDetector()
    results = detector.train_and_detect(raw_data)
    
    anomalies = results[results['is_anomaly'] == True]
    if not anomalies.empty:
        logging.warning(f"CRITICAL: {len(anomalies)} cost anomalies detected!")
        print(anomalies[['day', 'spend']])
    else:
        logging.info("No cost anomalies detected.")