from torch.utils.data import dataset
import os
from glob import glob
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import cv2
from PIL import Image
import scipy.misc as sci


class Dataset(dataset.Dataset):

    def __init__(self, root_dir, size=128, mode='train'):
        """
        Args:
            root_dir (string): Directory with all the data.
            mode (string): test or train
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mode = mode
        self.root_dir = root_dir
        self.datalist = glob(os.path.join(root_dir+mode+'data/', "*.npy"))
        self.segmlist = glob(os.path.join(root_dir + mode+'segm/', "*.npy"))
        self.size = size

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        image = np.load(self.datalist[idx]).astype(np.float32)
        segm = np.load(self.segmlist[idx]).astype(np.longlong)
        datamax = np.max(image)
        datamin = np.min(image)
        image = (image - datamin) / (datamax - datamin)
        k = random.uniform(0.9, 1.1)
        b = random.uniform(-0.1, 0.1)
        image = image * k + b
        image = np.clip(image, 0, 1)
        image = image.reshape((1, self.size, self.size))

        return image, segm
