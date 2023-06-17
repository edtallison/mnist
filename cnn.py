import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import MNIST
import wandb

train_loader = torch.utils.data.DataLoader(
    MNIST('./data', train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,)) # mean and std of MNIST
            ])),
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    MNIST('./data', train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])),
    batch_size=64, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1), # 1 input channel, 32 output channels, 5x5 kernel, stride 1
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4*4*64, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)
    
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train():
    model.train()
    for epoch in range(10):
        for batch_idx, (data, target) in enumerate(train_loader):
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print("Train Epoch: {}, iteration: {}, Loss: {}".format(
                    epoch, batch_idx, loss.item()
                ))
                wandb.log({"Train Loss": loss.item()})
        
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print("Test Loss: {}, Test Accuracy: {}".format(
        test_loss, test_accuracy
    ))
    wandb.log({"Test Loss": test_loss, "Test Accuracy": test_accuracy})

if __name__ == "__main__":
    wandb.init(project="mnist")
    train()
    test()

    torch.save(model.state_dict(), "models/mnist_cnn.pt")
    wandb.save("models/mnist_cnn.pt")