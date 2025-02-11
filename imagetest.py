!pip install --upgrade wandb
import wandb
wandb.login(key="eefffc915d794ad4ed436f057be395c1cc8108e7")
wandb.finish()
wandb.init(
    project="imagetest",  # Change project name as needed
    entity="manishb9",  # Replace with your WandB username
    config={
        "lr": 0.001,
        "epochs": 3,
        "batch_size": 64,
        "model_type": "CNN",
    }
)

config = wandb.config  # Retrieve experiment config
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Transform: Convert images to tensors and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load training & test data
train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def train_model(config):
    wandb.init(project="image-classification", config=config)  # Initialize Wandb inside the function
    config = wandb.config  # Load updated config

    model = SimpleCNN()  # Initialize the model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Initialize a table to store evaluation data
    table = wandb.Table(columns=["Image", "True Label", "Predicted Label"])

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                wandb.log({"training_loss": running_loss / 100, "epoch": epoch + 1})
                running_loss = 0.0

        # Evaluate Model
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Add images and predictions to the table during evaluation
                for i in range(images.size(0)):
                    img = wandb.Image(images[i])  # Convert the image tensor to a Wandb Image
                    true_label = labels[i].item()
                    pred_label = predicted[i].item()
                    table.add_data(img, true_label, pred_label)

        # Log accuracy and table to Wandb after evaluation
        accuracy = 100 * correct / total
        wandb.log({"test_accuracy": accuracy, "epoch": epoch + 1, "Live Predictions": table})

    # Save model locally
    model_path = "model.pth"
    torch.save(model.state_dict(), model_path)

    # Log model as an artifact
    artifact = wandb.Artifact("simple_cnn_model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    wandb.finish()  # Properly close Wandb run

# Start training
train_model(config)


sweep_config = {
    "method": "grid",  # Options: "random", "grid", "bayes"
    "metric": {"name": "test_accuracy", "goal": "maximize"},
    "parameters": {
        "lr": {"values": [0.001, 0.0005, 0.0001]},
        "epochs": {"values": [5, 10]},
    },
}
sweep_id = wandb.sweep(sweep_config, project="image-classification")


def sweep_train():
    with wandb.init() as run:
        config = run.config  # Load sweep-config parameters

        model = SimpleCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.lr)

        best_accuracy = 0.0  # Track the highest accuracy
        best_model_path = "best_model.pth"  # File to store the best model

        for epoch in range(config.epochs):
            model.train(True)
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    wandb.log({"training_loss": running_loss / 100, "epoch": epoch + 1})
                    running_loss = 0.0

            # Validation phase
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            wandb.log({"test_accuracy": accuracy, "epoch": epoch + 1})

            # Save model if it's the best so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), best_model_path)  # Save locally

        # Log the best model as a Wandb artifact
        artifact = wandb.Artifact("best_model", type="model")
        artifact.add_file(best_model_path)
        wandb.log_artifact(artifact)

        wandb.finish()  # Close Wandb properly
        wandb.agent(sweep_id, function=sweep_train, count=3)  # Runs 3 experiments