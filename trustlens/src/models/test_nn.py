# Quick test for the MLPClassifier neural network
import numpy as np
from sklearn.neural_network import MLPClassifier

print("1. Imports OK")

X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)
print("2. Dummy data OK")

model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=20, random_state=42)
model.fit(X, y)
print(f"3. Training OK — {model.n_iter_} iterations")

preds = model.predict(X[:5])
proba = model.predict_proba(X[:5])
print(f"4. Predictions OK — {preds}, proba shape: {proba.shape}")

print("5. ALL GOOD")
