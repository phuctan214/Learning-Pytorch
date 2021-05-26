import numpy as np
import torch
import torch.nn as nn


# def self_attention_basic(x):
#     raw_weights = torch.bmm(x, x.transpose(1, 2))
#     weight = F.softmax(raw_weights, dim=2)
#     return torch.bmm(weight, x)

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, query, key, value, mask=None):
        batch, head, seq_leg, d_model = key.size()

        key_transpose = key.view(batch, key, d_model, seq_leg)

        score = torch.mm(query, key_transpose)
        score = score / np.sqrt(d_model)
        # if mask:
        #     score = score.masked_fill
        score = self.softmax(score)

        return torch.mm(score, value)


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SelfAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.toquery = nn.Linear(d_model, d_model * n_heads, bias=False)
        self.tokey = nn.Linear(d_model, d_model * n_heads, bias=False)
        self.tovalue = nn.Linear(d_model, d_model * n_heads, bias=False)

        self.last_linear = nn.Linear(d_model * n_heads, d_model)
        self.scale_dot_product_attention = ScaleDotProductAttention()

    def forward(self, x):
        batch, seq_length, d_model = x.size()

        assert d_model == self.d_model, f'Input embedding dim ({d_model}) should match layer embedding dim ({self.d_model})'

        output_query = self.toquery(x)
        output_key = self.tokey(x)
        output_value = self.tovalue(x)

        output_query = output_query.view(batch, self.n_heads, seq_length, d_model)
        output_key = output_key.view(batch, self.n_heads, seq_length, d_model)
        output_value = output_value.view(batch, self.n_heads, seq_length, d_model)

        output_scale_dot_attention = self.scale_dot_product_attention(output_query, output_key, output_value)

        result = self.last_linear(output_scale_dot_attention)

        return result
