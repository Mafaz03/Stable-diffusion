import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPembedding(nn.Module):
    def __init__(self, n_vocab: int, n_embed: int, n_token: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.positioning_embed = nn.Parameter(torch.zeros(n_token, n_embed))


    def forward(self, tokens):

        # (batch_size, seq_len) -> (batch_size, seq_len, dim)
        x = self.token_embedding(tokens) 

        x += self.positioning_embed # (batch_size, seq_len, dim)

        return x
    
class CLIPlayer(nn.Module):
    def __init__(self, n_heads: int, n_embed: int):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_heads, n_embed)
        self.layer_norm_2 = nn.LayerNorm(n_embed)
        self.layer_1 = nn.Linear(n_embed, n_embed * 4)
        self.layer_2 = nn.Linear(n_embed * 4, n_embed)

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.layer_norm_1(x)
        x = self.attention(x, casual_mask=True)
        x += residual

        residual = x
        x = self.layer_norm_2(x)
        x = self.layer_1(x)

        x = x * torch.sigmoid(1.702 * x) # QuickGelu
        x = self.layer_2(x)
        x += residual

        return x


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPembedding(n_vocab=49408, n_embed=768, n_token=77)

        self.layers = nn.ModuleList([
            CLIPlayer(12, 768) for i in range(12)
        ])

        self.layer_norm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor):
        
        tokens = tokens.type(torch.long)
        state = self.embedding(tokens) # (batch_size, seq_len, dim)

        for layer in self.layers: 
            # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
            state = layer(state)
        output = self.layer_norm(state)
        
        return output # (Batch_Size, Seq_Len, Dim)

## Testing
if __name__ == "__main__":
    print("CLIP test: ")
    tokens = torch.rand(1,77)

    clip = CLIP()
    result = clip(tokens)
    print(result.shape) # [1, 77, 768]
