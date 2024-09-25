import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE(nn.Sequential):
    def __init__(self):
        super().__init__(
            ## Initial 
            # (batch_size, channels=3, 512, 512)
            nn.Conv2d(3, 128, kernel_size=3, padding=1), # (batch_size, 128, 512, 512)

            VAE_ResidualBlock(128, 128), # (batch_size, 128, 512, 512)
            VAE_ResidualBlock(128, 128), # (batch_size, 128, 512, 512)

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0), # (batch_size, 128, 256, 256)

            VAE_ResidualBlock(128, 256), # (batch_size, 256, 256, 256)
            VAE_ResidualBlock(256, 256), # (batch_size, 256, 256, 256)

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), # (batch_size, 256, 128, 128)

            VAE_ResidualBlock(256, 512), # (batch_size, 512, 128, 128)
            VAE_ResidualBlock(512, 512), # (batch_size, 512, 128, 128)

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), # (batch_size, 512, 64, 64)

            VAE_ResidualBlock(512, 512), # (batch_size, 512, 64, 64)
            VAE_ResidualBlock(512, 512), # (batch_size, 512, 64, 64)

            VAE_ResidualBlock(512, 512), # (batch_size, 512, 64, 64)

            VAE_AttentionBlock(512), # (batch_size, 512, 64, 64)

            VAE_ResidualBlock(512, 512), # (batch_size, 512, 64, 64)

            nn.GroupNorm(32, 512), # (batch_size, 512, 64, 64)

            nn.SiLU(),

            nn.Conv2d(512, 8, kernel_size=3, padding=1), # (batch_size, 8, 64, 64)
            nn.Conv2d(8, 8, kernel_size=1, padding=0), # (batch_size, 8, 64, 64)
        )

    def forward(self, x, noise):
        for module in self:
            print(module)
            if getattr(module, "stride", None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        
        # (batch_size, 8, 64, 64) -> (batch_size, 4, 64, 64) for mean and log_var
        mean, log_var = torch.chunk(x, 2, dim=1)

        # Clamp log_var instead of x
        log_var = torch.clamp(log_var, -30, 20)

        # Compute variance and std
        var = log_var.exp()
        std = var.sqrt()

        # Correct reparameterization trick: mean + std * noise
        # z shape -> (batch_size, 4, 64, 64)
        z = mean + std * noise
        z *= 0.18215

        return z
    
if __name__ == "__main__":
    x = torch.rand(1, 3, 512, 512)
    noise = torch.rand(1, 4, 64, 64)  # Adjusted noise to match the mean/log_var size
    vae = VAE()
    result = vae(x, noise)
    print(result.shape)
