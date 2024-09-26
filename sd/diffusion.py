import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class TimeEmbedding():
    pass
class UNET():
    pass
class UNNET_OutputLayer():
    pass

class diffusion(nn.Module):

    def __init__(self):
        super().__init__()

        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNNET_OutputLayer(320, 4)
    
    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        
        time = self.time_embedding(time) # (1,320) -> (1, 1280)

        output = self.unet(latent, context, time) # (batch_size, 4, 64, 64) -> (batch_size, 320, 64, 64)

        output = self.final(output) # (batch_size, 320, 64, 64) -> (batch_size, 4, 64, 64)

        return output
