from torch.utils.data import dataset
import os
from glob import glob
import numpy as np
import random


class Dataset(dataset.Dataset):

    def __init__(self, root_dir, size=128, mode='train'):
        """
        Args:
            root_dir (string): Directory with all the images.
            mode (string): test or train
        """
        self.mode = mode
        self.root_dir = root_dir
        self.datalist = glob(os.path.join(root_dir+mode+'data/', "*.npy"))
        self.size = size

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        image = np.load(self.datalist[idx]).astype(np.float32)
        flipw = random.random()
        if flipw < 0.5:
            image = image[:, ::-1]

        #####################################
        datamax = np.max(image)
        datamin = np.min(image)
        image = (image - datamin) / (datamax - datamin)
        datamean = np.sum(image) / (self.size * self.size)
        oriimg = image.copy()
        mask = np.zeros_like(image)
        for i in range(len(image)):
            ifzero = random.random()
            if ifzero < 0.5:
                mask[:, i] = 1
                image[:, i] = datamean

        #######################################
        image = image.reshape((1, self.size, self.size))
        oriimg = oriimg.reshape((1, self.size, self.size))
        mask = mask.reshape((1, self.size, self.size))
        return image, oriimg, mask
