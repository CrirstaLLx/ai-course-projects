import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Definovanie transformácií na predspracovanie dát
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Načítanie dát
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# Definovanie modelu
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Rozvinutie vstupu
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# Funkcie na trénovanie a vyhodnocovanie
def train_model(optimizer, model, criterion, train_loader, test_loader, epochs=10):
    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

        train_losses.append(total_loss / len(train_loader))
        test_accuracy = evaluate_model(model, test_loader)
        test_accuracies.append(test_accuracy)

        print(
            f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}, Accuracy: {correct / len(train_loader.dataset)}')

    return train_losses, test_accuracies


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
    accuracy = correct / len(test_loader.dataset)
    print(f'Test Accuracy: {accuracy}')
    return accuracy


# Funkcia na kreslenie grafov
def plot_results(train_losses, test_accuracies, title):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.title(f'{title} - Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.title(f'{title} - Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Funkcia na vytvorenie a vykreslenie confusion matrix
def plot_confusion_matrix(model, test_loader):
    model.eval()
    all_preds = torch.tensor([])
    all_labels = torch.tensor([])

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds = torch.cat((all_preds, preds), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


# Trénovanie s rôznymi optimizerami a kreslenie grafov
criterion = nn.CrossEntropyLoss()
epochs = 10

# SGD
model_sgd = MLP()
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
print("Training using SGD")
train_losses, test_accuracies = train_model(optimizer_sgd, model_sgd, criterion, train_loader, test_loader, epochs)
plot_results(train_losses, test_accuracies, "SGD")

# SGD with momentum
model_sgd_momentum = MLP()
optimizer_sgd_momentum = optim.SGD(model_sgd_momentum.parameters(), lr=0.01, momentum=0.9)
print("Training using SGD with momentum")
train_losses, test_accuracies = train_model(optimizer_sgd_momentum, model_sgd_momentum, criterion, train_loader,
                                            test_loader, epochs)
plot_results(train_losses, test_accuracies, "SGD with Momentum")

# Adam
model_adam = MLP()
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
print("Training using Adam")
train_losses, test_accuracies = train_model(optimizer_adam, model_adam, criterion, train_loader, test_loader, epochs)
plot_results(train_losses, test_accuracies, "Adam")
