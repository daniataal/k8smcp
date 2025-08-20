# model_definitions.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Define a dictionary to hold all model configurations
MODEL_CONFIGURATIONS = {
    "mnist_pytorch_cnn": {
        "model_name": "MNISTNet",
        "model_class_code": """class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)""",
        "model_load_logic": """model = {model_name}()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()""",
        "transform_code": """transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])""",
        "train_script_template": """import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Define CNN model for MNIST
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load training data
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Download and load test data
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000)

# Initialize model, optimizer and loss function
model = MNISTNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training function
def train(epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}')

# Testing function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

# Main function to run training and save the model
def main():
    epochs = 1
    print("Starting training...")
    train(epochs)
    print("Training finished.")

    print("Starting testing...")
    test()
    print("Testing finished.")
    
    # DVC: Pull data before training
    print("DVC: Pulling data...")
    os.system("dvc pull")

    # Save the model
    os.makedirs('/mnt/model', exist_ok=True)
    torch.save(model.state_dict(), '/mnt/model/mnist_cnn.pt')
    print("Model saved to /mnt/model/mnist_cnn.pt")

    # DVC: Add and push the trained model
    print("DVC: Adding model to DVC...")
    os.system("dvc add /mnt/model/mnist_cnn.pt")
    print("DVC: Pushing data and model...")
    os.system("dvc push")

if __name__ == '__main__':
    main()
""",
        "dockerfile_train_template": """FROM pytorch/pytorch:latest
WORKDIR /app
COPY . .
CMD ["python", "train.py"]
""",
        "dockerfile_infer_template": """FROM pytorch/pytorch:latest
WORKDIR /app
COPY . .
EXPOSE 8080
# Install prometheus_client
RUN pip install prometheus_client flask
CMD ["python", "inference.py"]
"""
    }
}
