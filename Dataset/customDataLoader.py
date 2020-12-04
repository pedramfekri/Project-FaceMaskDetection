import torch
# from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


class CustomDataLoader(Dataset):
    """CustomDataLoader"""

    def __init__(self, root_dir, dataframe, transform=None):
        """
        Args:

        """
        self.root_dir = root_dir
        self.transform = transform
        self.df = dataframe

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.df.iloc[idx, 0])

        # image = io.imread(img_name)
        image = Image.open(img_name)
        # print(image)
        if self.transform:
            image = self.transform(image)
        label = self.df.iloc[idx, 1:]
        label = np.array([label])
        label = label.astype('int').reshape(-1, 1)
        # sample = {'image': image, 'label': label}
        sample = [image, label]

        # if self.transform:
        #     sample = self.transform(sample)

        return sample