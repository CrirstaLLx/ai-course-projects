import matplotlib.pyplot as plt

def plot_loss(losses, title="Training loss"):
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()
