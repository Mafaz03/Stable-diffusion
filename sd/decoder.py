import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x: torch.Tensor):
        residual = x
        n, c, h, w = x.shape

        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2) # (n, h*w, c)

        x = self.attention(x)

        x = x.transpose(-2, -1)
        x = x.view((n, c, h, w))

        return x

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)

        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.resudual_layer = nn.Identity()
        else:
            self.resudual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x):

        residual = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.resudual_layer(residual)
    
class VAE_decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0), # (batch_size, 4, 64, 64)
            nn.Conv2d(4, 512, kernel_size=3, padding=1), # (batch_size, 512, 64, 64)

            VAE_ResidualBlock(512, 512), # (batch_size, 512, 64, 64)

            VAE_AttentionBlock(512), # (batch_size, 512, 64, 64))
                               
            VAE_ResidualBlock(512, 512), # (batch_size, 512, 64, 64)
            VAE_ResidualBlock(512, 512), # (batch_size, 512, 64, 64)
            VAE_ResidualBlock(512, 512), # (batch_size, 512, 64, 64)
            VAE_ResidualBlock(512, 512), # (batch_size, 512, 64, 64)

            nn.Upsample(scale_factor=2), # (batch_size, 512, 128, 128)

            nn.Conv2d(512, 512, kernel_size=3, padding=1), # (batch_size, 512, 128, 128)

            VAE_ResidualBlock(512, 512), # (batch_size, 512, 128, 128)
            VAE_ResidualBlock(512, 512), # (batch_size, 512, 128, 128)
            VAE_ResidualBlock(512, 512), # (batch_size, 512, 128, 128)

            nn.Upsample(scale_factor=2), # (batch_size, 512, 256, 256)

            VAE_ResidualBlock(512, 256), # (batch_size, 512, 256, 256)
            VAE_ResidualBlock(256, 256), # (batch_size, 512, 256, 256)
            VAE_ResidualBlock(256, 256), # (batch_size, 512, 256, 256)

            nn.Upsample(scale_factor=2), # (batch_size, 256, 512, 512)

            nn.Conv2d(256, 256, kernel_size=3, padding=1), # (batch_size, 256, 512, 512)

            VAE_ResidualBlock(256, 128), # (batch_size, 128, 512, 512)
            VAE_ResidualBlock(128, 128), # (batch_size, 128, 512, 512)
            VAE_ResidualBlock(128, 128), # (batch_size, 128, 512, 512)

            nn.GroupNorm(32, 128), # (batch_size, 128, 512, 512)

            nn.SiLU(), # (batch_size, 128, 512, 512)

            nn.Conv2d(128, 3, kernel_size=3, padding=1), # (batch_size, 3, 512, 512)

        )
    def forward(self, x: torch.Tensor):

        x /= 0.18215

        for module in self:
            x = module(x)
        
        # (batch_size, 3, 512, 512)
        return x
        



## Testing
if __name__ == "__main__":
    print("Residual Block Test: ")
    x = torch.rand(1, 128, 128, 128)
    vae_residual = VAE_ResidualBlock(128, 128)
    result = vae_residual(x)
    print(result.shape)

    x = torch.rand(1, 128, 128, 128)
    vae_residual = VAE_ResidualBlock(128, 256)
    result = vae_residual(x)
    print(result.shape)

    print("\nSelf Attention Block Test: ")
    x = torch.rand(4, 256, 16, 16)
    self_attention = VAE_AttentionBlock(256)
    result = self_attention(x)
    print(result.shape)

    print("\n VAE Decoder test: ")
    x = torch.rand(1, 4, 64, 64)
    decoder = VAE_decoder()
    result = decoder(x)
    print(result.shape)
