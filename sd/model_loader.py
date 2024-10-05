import torch
from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion

import model_converter

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }


def preload_models_from_my_weights(ckpt_path, diffuser_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)
    state_dict_diffuser = torch.load(diffuser_path)  # Adjust the path accordingly

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    # Load the state dictionary into the model

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict_diffuser, strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }


## Testing
if __name__ == "__main__":
    print("Testing loading pretrained weights: ")
    preload_models_from_standard_weights("data/v1-5-pruned-emaonly.ckpt", "cpu")
    print("Successfull import")