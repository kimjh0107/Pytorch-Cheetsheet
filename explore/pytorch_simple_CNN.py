# %%
import torch 
import torch.nn as nn # all neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions 
import torch.optim as optim # all optimization, SGD, Adam, etc 
import torch.nn.functional as F # all function that don't have any parameters 
from torch.utils.data import DataLoader  # gives easier datasets management and creates mini batches 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms

# %%
# Create Simple Fully Connected Neural Network 
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 

# Create simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1)) 
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) # 28 -> 14 
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1)) 
        self.fc1 = nn.Linear(16*7*7, num_classes)  # 16 = out_channels, 7 = input(28) 나누기 2번 by maxpooling 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1) # mini batch만 남겨두고 reshape 
        x = self.fc1(x)
        return x 


# %%
# Set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters 
in_channel = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# Load Data
train_dataset = datasets.MNIST(root = "../data/", train = True, transform = transforms.ToTensor(), download = True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root = "../data/", train = False, transform = transforms.ToTensor(), download = True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network 
model = CNN().to(device)

# Loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Train Network 
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader): 
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        # Get to correct shape -> 이미 앞에서 reshape를 진행함 
        # data = data.reshape(data.shape[0], -1) 
        
        # forward 
        scores = model(data)
        loss = criterion(scores, targets)
        # backward 
        optimizer.zero_grad() 
        loss.backward()
        # gradient descent or adam step 
        optimizer.step()

# %%
# Check accuracy on training & test to see how good our model 
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            #x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)  
            num_correct += (predictions == y).sum() 
            num_samples += predictions.size(0)  

    model.train()
    return num_correct/num_samples
# %%
print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")
# %%
