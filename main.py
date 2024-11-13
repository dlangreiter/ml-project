import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision

transforms = transforms.Compose([
    transforms.ToTensor()
])


data_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
data_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms)

data_train_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=100, shuffle=True)
data_test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=100, shuffle=False)


class NNNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NNNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.re1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.re1(x)
        x = self.fc2(x)
        return x
    

model = NNNet(28*28, 128, 10)
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.001)

epochs = 5

for i in range(epochs):
    for batch_id , (images,labels) in enumerate(data_train_loader):
        images = images.reshape(-1,28*28)
        output = model(images)
        loss = criterion(output,labels)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if (batch_id + 1) % 100 == 0:
            print(f'Epoch [{i+1}/{epochs}]')


torch.save(model.state_dict(), 'model.pth')