import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
import config

def unnormalize(tensor):
    # Unnormalize the image tensor (reverse of normalization)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def show_some(dl):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for image, descriptions in dl: # First batch
        for j in range(config.TRAIN_BATCH_SIZE):
            if j == 4: break

            inv_norm = unnormalize(image[j].clone())

            axs[j].imshow(inv_norm.permute(1,2,0))
            axs[j].set_axis_off()
            axs[j].set_title(descriptions[j])

        break # only first batch
    fig.show()

def generate_context(tokenizer, clip, prompt: list, uncond_prompt: list = []):

    # Convert into a list of length Seq_Len=77
    cond_tokens = tokenizer.batch_encode_plus(
        [*prompt], padding="max_length", max_length=77
    ).input_ids
    # (Batch_Size, Seq_Len)
    cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device="cpu")
    # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
    cond_context = clip(cond_tokens)
    # Convert into a list of length Seq_Len=77
    if uncond_prompt != []:
        uncond_tokens = tokenizer.batch_encode_plus(
            [*uncond_prompt], padding="max_length", max_length=77
        ).input_ids
    else: 
        uncond_tokens = tokenizer.batch_encode_plus(
            [""], padding="max_length", max_length=77
        ).input_ids
    # (Batch_Size, Seq_Len)
    uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device="cpu")
    # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
    uncond_context = clip(uncond_tokens)
    # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
    context = torch.cat([cond_context, uncond_context])

    return context

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.EVAL_BATCH_SIZE,
        generator=torch.manual_seed(5),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.OUTPUT_DIR, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")