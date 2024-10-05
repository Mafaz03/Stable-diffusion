from torchvision import transforms
import torch

IMAGE_SIZE = 64
LATENT_SIZE = IMAGE_SIZE//8
TRAIN_BATCH_SIZE = 4   
EVAL_BATCH_SIZE = 4   
NUM_TRAIN_STEPS = 1000 
EPOCHS = 50
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4
LR_WARMUP_STEPS = 500
SAVE_IMAGE_EPOCHS = 10
SAVE_MODEL_EPOCHS = 10
MIXED_PRECISION = "fp16"
OUTPUT_DIR = "data/diffusion.pth"
OVERWRITE_OUTPUT_DIR = True
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IDLE_DEVICE = "cpu"


initial_transforms = transforms.Compose([
                        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ])