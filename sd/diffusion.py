import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

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



class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time = 1280):
        super().__init__()

        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.group_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.resudual_layer = nn.Identity()
        else:
            self.resudual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, feature, time):
        
        # time (1, 1280)

        residual = feature
        feature = self.groupnorm_1(feature)
        feature = F.silu(feature)
        feature = self.conv_1(feature)
        
        time = F.silu(time)
        time = self.linear_time(time).unsqueeze(-1).unsqueeze(-1)

        merged = feature + time

        merged = self.group_merged(merged)
        merged = F.silu(feature)
        merged = self.conv_merged(feature)

        return merged + self.resudual_layer(residual)


class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_heads, n_embed, d_context = 768):
        super().__init__()

        channels = n_embed * n_heads

        self.group_norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layer_norm_1 = nn.LayerNorm(channels)
        self.self_attention_1 = SelfAttention(n_heads, channels)

        self.layer_norm_2 = nn.LayerNorm(channels)
        self.cross_attention_2 = CrossAttention(n_heads, channels, d_context, inprojbias = False)

        self.layer_norm_3 = nn.LayerNorm(channels)

        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        
        residual_long = x
        x = self.group_norm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape
        
        # (batch_size, channels, height, width)
        x = x.view(n, c, h*w)    # (batch_size, channels, height * width)
        x = x.transpose(-1, -2)  # (batch_size, height * width, channels)

        ## SelfAttention

        residual_short = x

        x = self.layer_norm_1(x)
        x = self.self_attention_1(x)
        x += residual_short

        ## CrossAttention

        residual_short = x

        x = self.layer_norm_2(x)
        x = self.cross_attention_2(x, context)
        x += residual_short

        ## Feedforward layers

        residual_short = x
        x = self.layer_norm_3(x)
        
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residual_short

        # (batch_size, height * width, channels)
        x = x.transpose(-1, -2)  # (batch_size, height, width, channels)
        x = x.view(n, c, h, w)    # (batch_size, channels, height,  width)
        

        return x + self.conv_output(residual_long)

        

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)
    

class UNNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x

class switchsequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x
    

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
            switchsequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),           # (batch_size, 640, 64, 64)
            switchsequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),                          # (batch_size, 320, 64, 64)
            switchsequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),                          # (batch_size, 320, 64, 64)
            switchsequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),                          # (batch_size, 320, 64, 64)
        ])

    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoder:
            x = layers(x, context, time)
            print(x.shape)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)
        print(x.shape)

        for layers in self.decoder:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            print(x.shape)
            x = layers(x, context, time)
        
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

## Testing

if __name__ == "__main__":
    print("Testing UNET: ")
    diff = diffusion()

    latent = torch.rand(1, 4, 64, 64)
    time = torch.rand(1, 320)
    context = torch.rand(1, 77, 768)

    result = diff(latent, context, time)
    print("\nResult Shape: ", result.shape)