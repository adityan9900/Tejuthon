import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from numpy import load

class NeuronDataset(Dataset):
    """MRI dataset."""

    def __init__(self, img_dir, label_dir, transform=None):
        """
        Args:
            vol_file (string): vol_file (.npz) contais the entire data volume
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

    #TODO: shuffle that damn data
    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_list = os.listdir(self.img_dir)
        label_list = os.listdir(self.label_dir)

        img = Image.open(os.path.join(self.img_dir, img_list[idx]))
        img = np.asarray(img)
        stacked_img = np.stack((img,)*3, axis=-1)
        img = Image.fromarray(stacked_img)

        label = Image.open(os.path.join(self.label_dir, label_list[idx]))
        label = np.asarray(label)
        stacked_label = np.stack((label,)*2, axis=-1)

        for i in range (stacked_label.shape[0]):
            for j in range (stacked_label.shape[1]):
                if(np.sum(stacked_label[i,j,:]) == 0): stacked_label[i,j,:] = [1,0]
                else: stacked_label[i,j,:] = [0,1]

        label = Image.fromarray(stacked_label)

        toTensor = transforms.ToTensor()
        label = toTensor(label)

        if self.transform:
            img = self.transform(img)

        return img, label
