# Workflow 
"""
1. import packages 
2. Create FCN
3. set device 
4. hyperparameters 
5. load data 
6. Initialize network 
7. Loss and optimizer 
8. Train Network 
9. Check accuracy on train test to see the result """

# %% 
import torch 
import torch.nn as nn  # inner network modeules 
import torch.optim as optim  # all the optimization algorithms 
import torch.nn.functional as F  # activation fucntion 
from torch.utils.data import DataLoader  
import torchvision.datasets as datasets 
import torchvision.transforms as transforms 

# %%
# Create Fully Connect Network 
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 

"""model = NN(784, 10)
x = torch.randn(64,784) # mini batch 를 사용할 경우 input으로 model에 사용가능 
x.shape
print(model(x).shape)   # torch.Size([64, 10])
"""
# %%
# Set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# Hyperparameters 
input_size = 784 
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# %%
# Load Data
train_dataset = datasets.MNIST(root = "../data/", train = True, transform = transforms.ToTensor(), download = True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root = "../data/", train = False, transform = transforms.ToTensor(), download = True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# %%
# Initialize network 
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# %%
# Train Network 
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader): # enumerate 사용 index와 원소 돌면서 루프 
                                                               # go throught each batch that we have in our training loader 
                                                               # data = img, targets = digit for each image 
        data = data.to(device=device)
        targets = targets.to(device=device)
       #print(data.shape)   # torch.Size([64, 1, 28, 28]), MNIST = 흑백 img 
        
        # Get to correct shape 
        data = data.reshape(data.shape[0], -1) # unroll this matrix into a long vector (first dimension just remain)
        
        # forward 
        scores = model(data)
        loss = criterion(scores, targets)

        # backward 
        optimizer.zero_grad()  # set all the gradients to 0 for each batch so that it doesn't store the back prop cal from previous forward prop
        loss.backward()

        # gradient descent or adam step 
        optimizer.step()

# %%
# Check accuracy on training & test to see how good our model 
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracty on the test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            # 64 x 10 -> maximum of 10 digits 구하기 
            _, predictions = scores.max(1)  # second dimensions scores are what we want 
            num_correct += (predictions == y).sum() # predict which is equally to labels 
            num_samples += predictions.size(0)  # batch = 64
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()
# %%
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
# %%
