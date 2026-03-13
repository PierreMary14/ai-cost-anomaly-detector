import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Mock cloud cost data (daily spend)
data = {
    'day': range(1, 31),
    'spend': [100, 102, 98, 105, 110, 100, 95, 100, 105, 500, # Spike at day 10
              110, 105, 100, 98, 102, 105, 100, 110, 105, 100,
              102, 98, 105, 110, 100, 95, 100, 105, 110, 105]
}

df = pd.DataFrame(data)

# AI Model: Isolation Forest for Anomaly Detection
model = IsolationForest(contamination=0.05)
df['anomaly'] = model.fit_predict(df[['spend']])

# Detect spikes (anomaly == -1)
anomalies = df[df['anomaly'] == -1]
print("FinOps AI Alert: Anomalies detected in cloud spend!")
print(anomalies)