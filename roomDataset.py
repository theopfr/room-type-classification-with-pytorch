
import numpy as np
import torchvision
import torch
import json
from PIL import Image
import matplotlib.pyplot as plt



class RoomDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, transform):
        self.dataset = json.load(open(file_path, "r"))
        self.transform = transform
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        image = np.array(Image.open(sample[0][0]).convert("RGB"))
        image = self.transform(image).view(3, image.shape[0], image.shape[1])

        label = torch.Tensor(sample[1])
            
        return image, label

    def __len__(self):
        return len(self.dataset)