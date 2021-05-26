import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#Hyperparameters
in_channel = 3
num_classes = 10
learning_rate = 0.001
batch_size = 1024
num_epochs = 5

#Load Pre-trained
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self,x):
        return x

model_pretrained = torchvision.models.vgg16(pretrained= True)

#Freeze All Parameters (Parameters Unchanged, No BackProp)
for para in model_pretrained.parameters():
    para.retains_grad = False

model_pretrained.avgpool = Identity()
model_pretrained.classifier = nn.Sequential(
    nn.Linear(512,100),
    nn.Dropout(p=0.5),
    nn.Linear(100,10)
)

model_pretrained.to(device)

# Load Data
train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)

test_dataset = datasets.MNIST(
    root="dataset/", train=False, transform=transforms.ToTensor(), download=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = model_pretrained

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Load Model
# if load_model:
#     load_checkpoint(torch.load("bi_lstm_checkpoint.pth.tar"))

#Train Network
for epoch in range(num_epochs):

    # if epoch % 2 == 1:
    #     checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    #     save_checkpoint(checkpoint)
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with \
              accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )
    # Set model back to train
    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)