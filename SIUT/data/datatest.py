from torch.utils.data import dataset
import numpy as np
import random


class Dataset(dataset.Dataset):

    def __init__(self, root_dir, datasize=128, rate=0.3):
        """
        Args:
            root_dir (string): './data/real_seismicdata/test/seismic2.npy'
            mode (string): test or train
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = np.load(root_dir)
        self.size = datasize
        stride = 120
        self.datamax = np.max(self.data)
        self.datamin = np.min(self.data)
        self.data = (self.data-self.datamin)/(self.datamax-self.datamin)
        self.data = self.data.astype(np.float32)
        # 对比度增强
        self.data = np.clip(self.data*1.5, 0, 1)

        self.label = self.data.copy()
        self.mask = np.zeros_like(self.data, np.float32)
        self.w, self.h = np.shape(self.data)
        # for j in range(self.h):
        #     ifdrop = random.random()
        #     if ifdrop<rate:
        #         self.data[:, j] = 0.5
        #         self.mask[:, j] = 1
        self.numw = self.w//stride
        self.numh = self.h//stride
        self.datalist=[]
        # self.labelist=[]
        # self.masklist = []

        for idw in range(self.numw):
            for idh in range(self.numh):
                subdata = self.data[idw*stride:idw*stride+datasize, idh*stride:idh*stride+datasize]
                # sublabel = self.label[idw * stride:idw*stride+datasize, idh * stride:idh*stride+datasize]
                # submask = self.mask[idw*stride:idw*stride+datasize, idh*stride:idh*stride+datasize]
                self.datalist.append({'locx':idw, 'locy':idh, 'data':subdata})
                # self.labelist.append({'locx': idw, 'locy': idh, 'data': sublabel})
                # self.masklist.append({'locx': idw, 'locy': idh, 'data': submask})
        # 最后一列
        for idh in range(self.numh):
            subdata = self.data[(self.w-datasize):self.w, idh*stride:idh*stride+datasize]
            # sublabel = self.label[(self.w-datasize):self.w, idh * stride:idh*stride+datasize]
            # submask = self.mask[(self.w-datasize):self.w, idh*stride:idh*stride+datasize]
            self.datalist.append({'locx':self.numw+1, 'locy':idh, 'data':subdata})
            # self.labelist.append({'locx': self.numw+1, 'locy': idh, 'data': sublabel})
            # self.masklist.append({'locx': self.numw+1, 'locy': idh, 'data': submask})
        # 最后一行
        for idw in range(self.numw):
            subdata = self.data[idw*stride:idw*stride+datasize, (self.h-datasize):self.h]
            # sublabel = self.label[idw*stride:idw*stride+datasize, (self.h-datasize):self.h]
            # submask = self.mask[idw*stride:idw*stride+datasize, (self.h-datasize):self.h]
            self.datalist.append({'locx': idw, 'locy': self.numh+1, 'data': subdata})
        #     self.labelist.append({'locx': idw, 'locy': self.numh+1, 'data': sublabel})
        #     self.masklist.append({'locx': idw, 'locy': self.numh+1, 'data': submask})
        # # 最后一块
        subdata = self.data[(self.w-datasize):self.w, (self.h - datasize):self.h]
        # sublabel = self.label[(self.w-datasize):self.w, (self.h - datasize):self.h]
        # submask = self.mask[(self.w-datasize):self.w, (self.h - datasize):self.h]
        self.datalist.append({'locx': self.numw+1, 'locy': self.numh + 1, 'data': subdata})
        # self.labelist.append({'locx': self.numw+1, 'locy': self.numh + 1, 'data': sublabel})
        # self.masklist.append({'locx': self.numw+1, 'locy': self.numh + 1, 'data': submask})


    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        image = self.datalist[idx]['data']
        image = image.reshape((1, self.size, self.size))
        # oriimg = self.labelist[idx]['data']
        # oriimg = oriimg.reshape((1, self.size, self.size))
        # mask = self.masklist[idx]['data']
        # mask = mask.reshape((1, self.size, self.size))
        return image   # , oriimg, mask
