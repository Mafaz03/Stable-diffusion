from tqdm import tqdm
from torch.utils.data import DataLoader
import config
from config import initial_transforms
from dataset import ImageDataset
import torch
from ddpm import DDPMSampler
import torch
from pipeline import rescale
from encoder import VAE_Encoder
from decoder import VAE_Decoder
import model_converter
from clip import CLIP
from utils import generate_context
from pipeline import get_time_embedding
from transformers import CLIPTokenizer
from torch.nn import functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
import gc


from diffusion import Diffusion
diffusion = Diffusion()
print("DIffuser model successfully initilised for training!")

generator = torch.Generator(device=config.DEVICE)
noise_scheduler = DDPMSampler(generator=generator)
print("DDPM model successfully initialised!")

clip = CLIP().to(config.DEVICE)
print("CLIP model successfully Loaded")


state_dict = model_converter.load_from_standard_weights("data/v1-5-pruned-emaonly.ckpt", config.DEVICE)

# Initialize the decoder
decoder = VAE_Decoder().to(config.DEVICE)
decoder.load_state_dict(state_dict['decoder'], strict=True)
print("Decoder model successfully Loaded")

# Initialize the encoder
encoder = VAE_Encoder().to()
encoder.load_state_dict(state_dict['encoder'], strict=True)  # This should load into the encoder
print("Encoder model successfully Loaded")

tokenizer = CLIPTokenizer("data/vocab.json", merges_file="data/merges.txt")
print("Tokensizer successfully Loaded")

train_ds = ImageDataset("sd/dataset/Train", initial_transforms, "sd/dataset/discription.json")
train_dl = DataLoader(train_ds, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)

test_ds = ImageDataset("sd/dataset/Test", initial_transforms, "sd/dataset/discription.json")
test_dl = DataLoader(train_ds, batch_size=config.EVAL_BATCH_SIZE, shuffle=True)

print("Dataset loaded Successfull!")


optimizer = torch.optim.AdamW(diffusion.parameters(), lr=config.LEARNING_RATE)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.LR_WARMUP_STEPS,
    num_training_steps=(len(train_dl) * config.EPOCHS),
)

print("Optimizer loaded and lr scheduler set!")

losses = []

global_step = 0

for epoch in range(config.EPOCHS):
    progress_bar = tqdm(total=len(train_dl))
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in enumerate(train_dl):
        clean_images = batch[0] # for now, didnt add the text description to images yet
        timestep = torch.randint(0, noise_scheduler.num_train_timesteps, (1,), device=clean_images.device) # One time step constant for each batch

        bs = clean_images.shape[0]
        latents_shape = (bs, 4, config.LATENT_SIZE, config.LATENT_SIZE)

        input_image_tensor = rescale(clean_images, (0, 255), (-1, 1))

        encoder_noise = torch.randn(latents_shape, generator=generator, device=clean_images.device)
        latents = encoder(input_image_tensor, encoder_noise)

        noise, noisy_latents = noise_scheduler.add_noise(latents, timestep)

        context = generate_context(tokenizer, clip, batch[1])
        context = torch.cat([context]*bs)

        time_embedding = get_time_embedding(timestep).to(clean_images.device.type)

        noise_pred_latent = diffusion(latents, context, time_embedding)

        loss = F.mse_loss(noise_pred_latent, noise)

        losses.append(loss)

        loss = loss / config.GRADIENT_ACCUMULATION_STEPS

        loss.backward()

        # Perform optimizer step only after accumulating enough gradients
        if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(train_dl):
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()
            lr_scheduler.step()

            # Zero the gradients
            optimizer.zero_grad()

        # Progress and logging
        progress_bar.update(1)
        logs = {
            "loss": loss.detach().item(),
            "lr": lr_scheduler.get_last_lr()[0],
            "step": global_step
        }
        progress_bar.set_postfix(**logs)

        # Update global step
        global_step += 1

    # CUDA out of memory, trying to tackle
    gc.collect()
    torch.cuda.empty_cache()

    if (epoch + 1) % config.SAVE_MODEL_EPOCHS == 0 or epoch == config.EPOCHS - 1:
        torch.save(diffusion.state_dict(), config.OUTPUT_DIR)