import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps= 1e-12):
        self.d_model = d_model
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self,x):
        mean = x.mean(-1, keepdim= True)
        std = x.std(-1, keepdim= True)

        x_hat = (x-mean)/(std +self.eps)
        return self.gamma * x_hat + self.beta

class FeedForward(nn.Module):
    def __init__(self,d_model, hidden_layer, prob_drop =0.2):
        self.d_model = d_model
        self.fc1 = nn.Linear(d_model,hidden_layer)
        self.fc2 = nn.Linear(hidden_layer,d_model)
        self.dropout= nn.Dropout(prob_drop)

    def forward(self,x):
        x =F.relu(self.fc1(x))
        x = self.dropout(x)

        return self.fc2(x)



