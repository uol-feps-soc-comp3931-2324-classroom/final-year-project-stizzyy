import matplotlib.pyplot as plt
import numpy as np
from albumentations import Compose, Resize, Normalize
import glob
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from utils.helpers import CLASS_DICT, get_labelled_mask
import config


# https://github.com/uygarkurt/UNet-PyTorch/blob/main/carvana_dataset.py

def parse_images(root_path, split):
    return {
        split : glob.glob(f'{root_path}/{split}/*.png'),
        'labels' : glob.glob(f'{root_path}/{split}_labels/*.png')
    }
        
def _transform(type):
    if type == 'image':
        return Compose([
            Resize(224, 224, always_apply=True),
            Normalize(
                mean=[0.45734706, 0.43338275, 0.40058118],
                std=[0.23965294, 0.23532275, 0.2398498],
                always_apply=True
            )
        ])
    elif type == 'label':
        return Compose([
            Resize(224, 224, always_apply=True)
        ])

class CamVid(Dataset):
    CLASS_DICT = CLASS_DICT

    def __init__(self, root_path, split, transform=True):
        self.root_path = root_path

        assert split in ['train', 'val']
        self.split = split
        self.files = parse_images(root_path, self.split)
        
        # handle transforms
        self.transform = transform
        self.image_transform = _transform('image')
        self.label_transform = _transform('label')

    def __len__(self):
        return len(self.files['labels'])
    
    def __getitem__(self, idx):
        image = np.array(Image.open(self.files[self.split][idx]).convert('RGB'))
        label = np.array(Image.open(self.files['labels'][idx]).convert('RGB'))

        if self.transform:
            image = self.image_transform(image=image)['image']
            label = self.label_transform(image=label)['image']

        # RGB mask -> ONE-HOT-LIKE encoded mask
        label = get_labelled_mask(label, CLASS_DICT)
            
        # image: HWC -> torch: CHW
        image = np.transpose(image, (2, 0, 1))

        image = torch.tensor(image, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        return image, label

train_dataset = CamVid(config.ROOT_PATH, 'train')
train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE)

val_dataset = CamVid(config.ROOT_PATH, 'val')
val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)