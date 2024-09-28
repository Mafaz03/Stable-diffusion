import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from ddpm import DDPMSampler


HEIGHT = 512
WIDTH = 512
LATENT_HEIGHT = HEIGHT // 8
LATENT_WIDTH = WIDTH // 8


def generator(prompt: str, uncond_promt: str, input_image=None, strenght: int = 0.8, do_cfg = True, 
              cfg_scale=7.5, sampler_name="ddpm", n_inference_steps = 50, models = (), seed = None,
              device = None, idle_device = None, tokenizer = None):
    
    with torch.no_grad():
        if not (0  < strenght <= 1): 
            raise ValueError("Strength must be in range of (0, 1)")
    
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)

        if seed is None: generator.seed()
        else: generator.manual_seed(seed)

        clip = models['clip']
        clip.to(device)

        if do_cfg:
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids # (batch_size, Seq_length)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            cond_context = clip(cond_tokens) # (batch_size, Seq_length, Dim)

            uncond_tokens = tokenizer.batch_encode_plus([uncond_promt], padding="max_length", max_length=77).input_ids # (batch_size, Seq_length)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens) # (batch_size, Seq_length, Dim)

            context = torch.cat([cond_context, uncond_context], dim=0) # (2, 77, 768)
        else:
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids # (batch_size, Seq_length)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            cond_context = clip(cond_tokens) # (batch_size, Seq_length, Dim): (2, 77, 768)
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else: raise ValueError(f"Unkown Sampler {sampler_name}")

        latent_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = torch.tensor(np.array(input_image_tensor), dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0) # (batch_size, width, height, channels)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2) # (batch_size, channels, width, height)

            encoder_noise = torch.rand((latent_shape), generator=generator, device=device)
            latent = encoder(input_image_tensor, encoder_noise)

            sampler.get_strength(strenght=strenght)
            sampler.add_noise(latent, sampler.timesteps[0])

            to_idle(encoder)

        else:
            latent = torch.rand(latent_shape, generator=generator, device=device)







