import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embed):
        super().__init__()

        self.liner_1 = nn.Linear(n_embed, 4 * n_embed)
        self.liner_2 = nn.Linear(4 * n_embed, 4 * n_embed)
    
    def forward(self, x):
        x = self.liner_1(x) # (1, 320) -> (1, 1280)
        x = F.silu(x)
        x = self.liner_2(x) # (1, 320) -> (1, 1280)
    
        return x

class switchsequential(nn.Sequential):
    def __init__(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        super().__init__()
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class UNET_AttentionBlock():
    pass
class UNET_ResidualBlock():
    pass

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.ModuleList([
            # (batch_size, 4, 64, 64)
            switchsequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),                 # (batch_size, 320, 64, 64)
            switchsequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),    # (batch_size, 320, 64, 64)
            switchsequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),    # (batch_size, 320, 64, 64)
            switchsequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),     # (batch_size, 320, 32, 32)
            switchsequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),    # (batch_size, 640, 32, 32)
            switchsequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),    # (batch_size, 640, 32, 32)
            switchsequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),     # (batch_size, 640, 16, 16)
            switchsequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),  # (batch_size, 1280, 16, 16)
            switchsequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)), # (batch_size, 1280, 16, 16)
            switchsequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),   # (batch_size, 1280, 8, 8)
            switchsequential(UNET_ResidualBlock(1280, 1280)),                              # (batch_size, 1280, 8, 8)
            switchsequential(UNET_ResidualBlock(1280, 1280)),                              # (batch_size, 1280, 8, 8)
        ])

        self.bottleneck = switchsequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),
        )

        self.decoder = nn.ModuleList([
            switchsequential(UNET_ResidualBlock(2*1280, 1280)),                                                  # (batch_size, 1280, 8, 8)
            switchsequential(UNET_ResidualBlock(2*1280, 1280)),                                                  # (batch_size, 1280, 8, 8)
            switchsequential(UNET_ResidualBlock(2*1280, 1280), Upsample(1280)),                                  # (batch_size, 1280, 16, 16)
            switchsequential(UNET_ResidualBlock(2*1280, 1280), UNET_AttentionBlock(8, 160)),                     # (batch_size, 1280, 16, 16)
            switchsequential(UNET_ResidualBlock(2*1280, 1280), UNET_AttentionBlock(8, 160)),                     # (batch_size, 1280, 16, 16)
            switchsequential(UNET_ResidualBlock(640+1280, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),   # (batch_size, 1280, 32, 32)
            switchsequential(UNET_ResidualBlock(640+1280, 640), UNET_AttentionBlock(8, 80)),                     # (batch_size, 640, 32, 32)
            switchsequential(UNET_ResidualBlock(2*640, 640), UNET_AttentionBlock(8, 80)),                        # (batch_size, 640, 32, 32)
            switchsequential(UNET_ResidualBlock(2*320, 640), UNET_AttentionBlock(8, 80), Upsample(640)),         # (batch_size, 640, 64, 64)
            switchsequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),                          # (batch_size, 640, 64, 64)
            switchsequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 80)),                          # (batch_size, 640, 64, 64)
            switchsequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),                          # (batch_size, 640, 64, 64)
        ])


class UNNET_OutputLayer():
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x


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
