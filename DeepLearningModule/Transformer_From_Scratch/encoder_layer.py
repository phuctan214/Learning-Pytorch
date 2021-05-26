from layer_norm import LayerNorm, FeedForward
from positional_embedding import PositionalEmbedding
from self_attention import SelfAttention
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, prob_drop, ff_hidden_layer, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.self_attention = SelfAttention(d_model=d_model, n_heads= n_heads)
        self.layer_norm = LayerNorm(d_model= d_model)
        self.feedforward = FeedForward(d_model=d_model, hidden_layer=ff_hidden_layer,prob_drop= prob_drop)

    def forward(self,x):
        x_ = self.self_attention(x)
        x_ = self.layer_norm(x_)
        output = x_+x
        output = self.feedforward(output)
        output_ = self.layer_norm(output)
        return output_ + output