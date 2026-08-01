import numpy as np

class MSELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true)**2)

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_pred.size
