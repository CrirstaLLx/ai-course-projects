import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.1
        self.b = np.zeros((1, out_features))

        self.x = None
        self.dW = None
        self.db = None

        # pre momentum
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        # dL/dW = X^T * dL/dY
        self.dW = self.x.T @ grad_output
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        # dL/dX = dL/dY * W^T
        return grad_output @ self.W.T

    def update(self, lr, momentum=0.0):
        # momentum
        self.vW = momentum * self.vW - lr * self.dW
        self.vb = momentum * self.vb - lr * self.db

        self.W += self.vW
        self.b += self.vb
