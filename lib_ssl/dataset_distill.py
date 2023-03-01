from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

class DistilledDataset(Dataset):

    def __init__(self,file_path,transform) -> None:
        super().__init__()
        LATENT_DIM = 3584
        IM_SIZE = 64
        data = torch.load(file_path)
        images = data[...,LATENT_DIM:].reshape(-1,3,IM_SIZE,IM_SIZE).permute(0,2,3,1).numpy()
        latent = data[...,:100].argmax(-1)
        self.images = images
        self.transform = transform
        self.labels = latent

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        
        img = self.images[index] * 0.22 + 0.45
        img = (np.clip(img,0,1)*255).astype(np.uint8)
        img = Image.fromarray(img)
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img,label