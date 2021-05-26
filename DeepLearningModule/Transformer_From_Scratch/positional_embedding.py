import numpy as np
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, dimension_model, max_len, device):
        super(PositionalEmbedding, self).__init__()
        self.dimension_model = dimension_model
        self.max_len = max_len

        self.positional_encoding = torch.zeros(max_len, dimension_model, device=device)
        self.positional_encoding.requires_grad = False  # Frozen variable

        for pos in range(max_len):
            for index in range(dimension_model):
                if index % 2 == 0:
                    self.positional_encoding[pos, index] = np.sin(pos / (1000 ** (2 * index / dimension_model)))
                else:
                    self.positional_encoding[pos, index] = np.cos(pos / (1000 ** (2 * index / dimension_model)))

    def forward(self, x):
        bacth, length = x.size()

        return self.positional_encoding[:length, :]


test_positional = PositionalEmbedding(dimension_model=1024, max_len=500, device='cpu')
variable = torch.rand((10, 30))
print(test_positional(variable))
