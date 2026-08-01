import numpy as np
from layers import Linear
from activations import Sigmoid, Tanh, ReLU
from losses import MSELoss
from model import SequentialModel
from utils import plot_loss


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

AND = np.array([[0], [0], [0], [1]])
OR  = np.array([[0], [1], [1], [1]])
XOR = np.array([[0], [1], [1], [0]])


def train(model, X, y, lr=0.1, momentum=0.0, epochs=500):
    loss_fn = MSELoss()
    losses = []

    for epoch in range(epochs):
        out = model.forward(X)
        loss = loss_fn.forward(out, y)
        grad = loss_fn.backward()
        model.backward(grad)
        model.update(lr, momentum)

        losses.append(loss)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss={loss:.4f}")

    return losses


if __name__ == "__main__":

    print("\n=== TRAINING XOR (2–4–1) ===")
    model1 = SequentialModel([
        Linear(2, 4),
        Tanh(),
        Linear(4, 1),
        Sigmoid()
    ])

    loss1 = train(model1, X, XOR, lr=0.1, momentum=0.0, epochs=500)
    plot_loss(loss1, "XOR – 2-4-1")

    print("\nPredictions:")
    print(model1.forward(X))


    print("\n=== TRAINING XOR (2–4–4–1) ===")
    model2 = SequentialModel([
        Linear(2, 4),
        Tanh(),
        Linear(4, 4),
        ReLU(),
        Linear(4, 1),
        Sigmoid()
    ])

    loss2 = train(model2, X, XOR, lr=0.1, momentum=0.3, epochs=500)
    plot_loss(loss2, "XOR – 2-4-4-1 with momentum 0.3")

    print("\nPredictions:")
    print(model2.forward(X))
