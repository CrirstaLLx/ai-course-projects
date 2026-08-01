import numpy as np
from model import SequentialModel
from layers import Linear
from activations import Sigmoid
from losses import MSELoss
from utils import plot_loss


AND_X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=float)

AND_Y = np.array([[0],
                  [0],
                  [0],
                  [1]], dtype=float)

OR_X = np.array([[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]], dtype=float)

OR_Y = np.array([[0],
                 [1],
                 [1],
                 [1]], dtype=float)


def train_dataset(X, Y, hidden_layers, lr=0.1, momentum=0.0, epochs=500, title=""):
    layers = []

    input_dim = X.shape[1]

    # Build network dynamically
    for h in hidden_layers:
        layers.append(Linear(input_dim, h))
        layers.append(Sigmoid())
        input_dim = h

    # Output layer
    layers.append(Linear(input_dim, 1))
    layers.append(Sigmoid())

    model = SequentialModel(layers)
    loss_fn = MSELoss()

    losses = []

    # Training loop
    for epoch in range(epochs):
        y_pred = model.forward(X)
        loss = loss_fn.forward(y_pred, Y)
        losses.append(loss)

        grad = loss_fn.backward()
        model.backward(grad)

        # Update parameters
        model.update(lr=lr, momentum=momentum)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss={loss:.4f}")

    # Plot
    plot_loss(losses, title)

    print("\nPredictions:")
    print(model.forward(X))


if __name__ == "__main__":

    print("\n=== TRAINING AND (2–4–1) ===")
    train_dataset(AND_X, AND_Y,
                  hidden_layers=[4],
                  lr=0.1,
                  momentum=0.0,
                  title="AND – 2-4-1")

    print("\n=== TRAINING OR (2–4–1) ===")
    train_dataset(OR_X, OR_Y,
                  hidden_layers=[4],
                  lr=0.1,
                  momentum=0.0,
                  title="OR – 2-4-1")

    print("\n=== TRAINING AND (2–4–4–1) ===")
    train_dataset(AND_X, AND_Y,
                  hidden_layers=[4, 4],
                  lr=0.1,
                  momentum=0.3,
                  title="AND – 2-4-4-1 with momentum 0.3")

    print("\n=== TRAINING OR (2–4–4–1) ===")
    train_dataset(OR_X, OR_Y,
                  hidden_layers=[4, 4],
                  lr=0.1,
                  momentum=0.3,
                  title="OR – 2-4-4-1 with momentum 0.3")
