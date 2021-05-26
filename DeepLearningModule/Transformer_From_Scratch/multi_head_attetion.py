from torch import nn
from einops import rearrange
import torch

from scale_dot_product_attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model*n_head)
        self.w_k = nn.Linear(d_model, d_model*n_head)
        self.w_v = nn.Linear(d_model, d_model*n_head)
        self.w_concat = nn.Linear(d_model*n_head, d_model)

    def forward(self, x, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        # 2. split tensor by number of heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_head), (q, k, v))

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        # out = self.concat(out)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.w_concat(out)

        return out

x = torch.rand((1,6,100))
multi_attention = MultiHeadAttention(d_model=100,n_head=2)
print(multi_attention(x).shape)