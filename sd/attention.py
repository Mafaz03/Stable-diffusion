import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bais = True, out_proj_bias = True):
        super().__init__()
        
        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias = in_proj_bais)
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x: torch.Tensor, casual_mask = True):
        input_shape = x.shape # (n, seq, Dim)

        batch_size, sequence_length, d_embed = input_shape

        interm_shape = (batch_size, sequence_length, self.n_heads, self.d_head) # (batch_size, seq, heads, Dim/heads)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(interm_shape).transpose(1, 2) # (batch_size, heads, seq, Dim/heads)
        k = k.view(interm_shape).transpose(1, 2) # (batch_size, heads, seq, Dim/heads)
        v = v.view(interm_shape).transpose(1, 2) # (batch_size, heads, seq, Dim/heads)

        weight = q @ k.transpose(-1, -2) # (batch_size, heads, seq, seq)

        if casual_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        output = weight @ v # (batch_size, heads, seq, Dim/heads)

        output = output.transpose(1, 2) # (batch_size, seq, heads, Dim/heads)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (n, seq, Dim)
        return output