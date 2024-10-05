import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias = True, out_proj_bias = True):
        super().__init__()
        
        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x: torch.Tensor, causal_mask = False):
        input_shape = x.shape # (n, seq, Dim)

        batch_size, sequence_length, d_embed = input_shape

        interm_shape = (batch_size, sequence_length, self.n_heads, self.d_head) # (batch_size, seq, heads, Dim/heads)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(interm_shape).transpose(1, 2) # (batch_size, heads, seq, Dim/heads)
        k = k.view(interm_shape).transpose(1, 2) # (batch_size, heads, seq, Dim/heads)
        v = v.view(interm_shape).transpose(1, 2) # (batch_size, heads, seq, Dim/heads)

        weight = q @ k.transpose(-1, -2) # (batch_size, heads, seq, seq)

        if causal_mask:
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
    
class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_context, in_proj_bias = True, out_proj_bias = True):
        super().__init__()

        self.q_proj = nn.Linear(d_embed, d_embed, bias = in_proj_bias)
        self.k_proj = nn.Linear(d_context, d_embed, bias = in_proj_bias)
        self.v_proj = nn.Linear(d_context, d_embed, bias = in_proj_bias)
        
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x: (latent): (batch_size, seq_length, dimentions (channels))
        # y: (batch_size, seq_length_KV, dimentions_KV): (batch_size, 77, 768)

        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape

        interm_shape = (batch_size, -1, self.n_heads, self.d_head) # (batch_size, seq, heads, Dim/heads)
        # print(interm_shape)

        q_proj = self.q_proj(x)
        k_proj = self.k_proj(y)
        v_proj = self.v_proj(y)

        q = q_proj.view(interm_shape).transpose(1, 2) # (batch_size, heads, seq, Dim/heads)
        # print(k_proj)
        k = k_proj.view(interm_shape).transpose(1, 2) # (batch_size, heads, seq, Dim/heads)
        v = v_proj.view(interm_shape).transpose(1, 2) # (batch_size, heads, seq, Dim/heads)

        # (batch_size, heads, seq, Dim/heads) x (batch_size, heads, Dim/heads, seq)
        weight = q @ k.transpose(-1, -2) # (batch_size, heads, seq, seq)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # (batch_size, heads, seq, seq) x (batch_size, heads, seq, Dim/heads)
        output = weight @ v # (batch_size, heads, seq, Dim/heads)

        output = output.transpose(1, 2) # (batch_size, seq, heads, Dim/heads)
        output = output.reshape(input_shape)

        output = self.out_proj(output)

        return output

## Testing

if __name__ == "__main__":
    print("Testing Self Attention Block: ")
    self_attention = SelfAttention(8, 256)
    result = self_attention(torch.rand(4, 256, 256))
    print(result.shape)

    print("\nTesting Cross Attention Block: ")
    corss_attention = CrossAttention(8, 768, 256)
    result = corss_attention(torch.rand(4, 768, 768), torch.rand(4, 768, 768))
    print(result.shape)
