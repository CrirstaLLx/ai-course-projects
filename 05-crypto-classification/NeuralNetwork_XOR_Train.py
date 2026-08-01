import numpy as np
import matplotlib.pyplot as plt


class LinearLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)  # He initialization
        self.bias = np.zeros((1, output_size))
        self.velocity_w = np.zeros_like(self.weights)
        self.velocity_b = np.zeros_like(self.bias)

    def forward(self, input_data):
        self.input = input_data
        return np.dot(input_data, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate, momentum=0.0):
        weights_gradient = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)

        # Applying momentum
        self.velocity_w = momentum * self.velocity_w + learning_rate * weights_gradient
        self.velocity_b = momentum * self.velocity_b + learning_rate * np.sum(output_gradient, axis=0, keepdims=True)

        # Update weights and biases
        self.weights -= self.velocity_w
        self.bias -= self.velocity_b

        return input_gradient


class ReLU:
    def forward(self, input_data):
        self.output = np.maximum(0, input_data)
        return self.output

    def backward(self, output_gradient):
        return output_gradient * (self.output > 0)


class Sigmoid:
    def forward(self, input_data):
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output

    def backward(self, output_gradient):
        return output_gradient * (self.output * (1 - self.output))


class Tanh:
    def forward(self, input_data):
        self.output = np.tanh(input_data)
        return self.output

    def backward(self, output_gradient):
        return output_gradient * (1 - self.output ** 2)


class MSE:
    def loss(self, prediction, target):
        return np.mean((prediction - target) ** 2)

    def backward(self, prediction, target):
        return 2 * (prediction - target) / target.size


def train_network(X_train, y_train, hidden_layers_sizes, learning_rate, epochs, use_momentum=False, momentum=0.0):
    layers = []
    activations = []

    input_size = X_train.shape[1]

    # Hidden layers with Tanh
    for hidden_size in hidden_layers_sizes:
        layers.append(LinearLayer(input_size, hidden_size))
        activations.append(Tanh())
        input_size = hidden_size

    # Output layer with Sigmoid
    layers.append(LinearLayer(input_size, 1))
    activations.append(Sigmoid())
    mse = MSE()

    losses = []

    for epoch in range(epochs):
        # Forward pass
        A = X_train
        for layer, activation in zip(layers, activations):
            A = layer.forward(A)
            A = activation.forward(A)

        prediction = A

        # Loss
        loss = mse.loss(prediction, y_train)
        losses.append(loss)

        # Backward pass
        gradient = mse.backward(prediction, y_train)

        for layer, activation in reversed(list(zip(layers, activations))):
            gradient = activation.backward(gradient)
            gradient = layer.backward(gradient, learning_rate, momentum=momentum if use_momentum else 0.0)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss}")

    # Plot loss curve
    plt.plot(losses)
    plt.title(f'Training Loss (hidden_layers={len(hidden_layers_sizes)}, lr={learning_rate}, momentum={momentum})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    return layers, activations


def test_network(X_test, y_test, layers, activations):
    print("Results:")
    for x, actual in zip(X_test, y_test):
        a = x
        for layer, activation in zip(layers, activations):
            a = layer.forward(a)
            a = activation.forward(a)
        predicted = a.flatten()
        print(f"Input: {x}, Predicted: {predicted[0]:.3f}, Target: {actual[0]}")


# Data for AND, OR, XOR
X_and_or_xor = {
    "AND": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [0], [0], [1]])),
    "OR": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [1]])),
    "XOR": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [0]]))
}

# Train and test network for each configuration
for problem, (X, y) in X_and_or_xor.items():
    for hidden_layers_sizes in [[4], [4, 4]]:
        for learning_rate in [0.01, 0.1]:
            for use_momentum in [False, True]:  # Try with and without momentum
                momentum_value = 0.9 if use_momentum else 0.0
                layers, activations = train_network(X, y, hidden_layers_sizes, learning_rate, epochs=500, use_momentum=use_momentum, momentum=momentum_value)
                print(
                    f"\nTraining for {problem} | Activation: {activations}, Learning rate: {learning_rate}, Momentum: {momentum_value}, Layers: {len(hidden_layers_sizes)}")
                print(f"Results for {problem} task:")
                test_network(X, y, layers, activations)
