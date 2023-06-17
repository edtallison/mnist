import torch
import torchvision
import torch.nn as nn
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


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)
    
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(data.shape)
        # print(target.shape)
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
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print("Test loss: {}, Accuracy: {}".format(
        test_loss, test_accuracy
    ))
    return test_loss, test_accuracy

wandb.init(project="mnist")
wandb.watch(model, log="all")

for epoch in range(1, 10):
    train(epoch)
    test_loss, test_accuracy = test()
    wandb.log({"Test Accuracy": test_accuracy, "Test Loss": test_loss})

torch.save(model.state_dict(), "models/mlp.pth")
wandb.save("mlp.pth")
