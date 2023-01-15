import numpy as np
from deep_data_depth_dennisweiss import DeepDataDepthAnomalyDetector

anomaly_detector = DeepDataDepthAnomalyDetector()

x = np.random.randn(50, 3)
y = np.zeros(50)
anomaly_detector.fit(x, y)
print(anomaly_detector.predict_score(x).shape[0])
