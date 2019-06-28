from torch.utils.data import dataset
import os
from glob import glob
import numpy as np


class Dataset(dataset.Dataset):

    def __init__(self, root_dir, size=128, mode='train'):
        """
        Args:
            root_dir (string): Directory with all the images.
            mode (string): test or train
        """
        self.mode = mode
        self.root_dir = root_dir
        self.datalist = glob(os.path.join(root_dir + mode + 'data/', "*.npy"))
        self.masklist = glob(os.path.join(root_dir + mode + 'datamask/', "*.npy"))
        self.size = size

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        image = np.load(self.datalist[idx]).astype(np.float32)
        mask = np.load(self.masklist[idx]).astype(np.float32)
        rmask = 1 - mask
        datamax = np.max(image)
        datamin = np.min(image)
        oriimg = (image - datamin) / (datamax - datamin)
        datamean = np.sum(oriimg) / (self.size * self.size)
        image = oriimg.copy()
        image = image * rmask + datamean * mask
        image = image.reshape((1, self.size, self.size))
        mask = mask.reshape((1, self.size, self.size))
        oriimg = oriimg.reshape((1, self.size, self.size))

        return image, oriimg, mask