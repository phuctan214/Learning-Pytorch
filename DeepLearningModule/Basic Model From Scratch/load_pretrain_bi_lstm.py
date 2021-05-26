import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 28
sequence_lenth = 28
num_layers = 2
num_classes = 10
hidden_size = 256
learning_rate = 0.001
batch_size = 64
num_epochs = 2
load_model = True

# Model Bidirectional LSTM
class BiRNN(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers, batch_first= True,
                            bidirectional= True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers *2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers *2, x.size(0), self.hidden_size).to(device)

        out, (hidden_state, cell_state) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return  out

#Function Load Checkpoint
def load_checkpoint(state):
    print("load checkpoint")
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])

#Function Save Checkpoint
def save_checkpoint(state, filename= "bi_lstm_checkpoint.pth.tar"):
    print("Saving checkpoint")
    torch.save(state,filename)



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
model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Load Model
if load_model:
    load_checkpoint(torch.load("bi_lstm_checkpoint.pth.tar"))

# Train Network
# for epoch in range(num_epochs):
#
#     if epoch % 2 == 1:
#         checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
#         save_checkpoint(checkpoint)
#     for batch_idx, (data, targets) in enumerate(train_loader):
#         # Get data to cuda if possible
#         data = data.to(device=device).squeeze(1)
#         targets = targets.to(device=device)
#
#         # forward
#         scores = model(data)
#         loss = criterion(scores, targets)
#
#         # backward
#         optimizer.zero_grad()
#         loss.backward()
#
#         # gradient descent or adam step
#         optimizer.step()

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


# check_accuracy(train_loader, model)
check_accuracy(test_loader, model)