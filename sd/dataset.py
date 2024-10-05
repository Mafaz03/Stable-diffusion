from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import json

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform, descriptions_file):
        super().__init__()
        self.transform = transform
        self.root_dir = root_dir
        self.image_list = os.listdir(root_dir)

        # Load the descriptions from a JSON file
        with open(descriptions_file, 'r') as f:
            self.descriptions = json.load(f)

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
    
        # Get the corresponding description for the image
        description = self.descriptions.get(image_name, "No description available.")
        
        return image, description