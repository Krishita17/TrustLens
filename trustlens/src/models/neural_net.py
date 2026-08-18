"""
Feedforward Neural Network for ZTA authentication decisions.
Uses sklearn MLPClassifier — fast, reliable, no PyTorch dependency issues.
"""

import logging
import numpy as np
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger(__name__)


class NeuralNetClassifier:
    """Sklearn MLPClassifier wrapper for ZTA authentication."""

    def __init__(self, input_dim: int = None, hidden_layers: list = None,
                 dropout: float = 0.3, lr: float = 0.001, epochs: int = 50,
                 batch_size: int = 256, patience: int = 10, random_state: int = 42,
                 verbose: bool = False):
        self.input_dim = input_dim
        self.hidden_layers = tuple(hidden_layers or [128, 64, 32])
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose
        self.classes_ = np.array([0, 1])

        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layers,
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=min(batch_size, 1024),
            learning_rate_init=lr,
            max_iter=epochs,
            early_stopping=True,
            n_iter_no_change=patience,
            validation_fraction=0.1,
            random_state=random_state,
            verbose=verbose,
        )

    def fit(self, X, y):
        X_arr = np.array(X)
        y_arr = np.array(y).ravel()
        logger.info(f"Training MLPClassifier: layers={self.hidden_layers}, max_iter={self.epochs}")
        self.model.fit(X_arr, y_arr)
        best = self.model.best_loss_ if self.model.best_loss_ is not None else self.model.loss_
        logger.info(f"Training complete: {self.model.n_iter_} iterations, best loss: {best:.4f}")
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(np.array(X))

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(np.array(X))
