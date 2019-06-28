import cv2
import os
import numpy as np
from utils.config import opt
from model.DeviceAdapt_Net import Ftrainer
import torch
import math


def test_model(testdata, mask, model, savedir):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    model.eval()
    h, w = testdata.shape
    scalemax = np.max(testdata)
    scalemin = np.min(testdata)
    np.save(savedir + 'input.npy', testdata * (1 - mask))
    np.save(savedir + 'label.npy', testdata)
    inputimg = (testdata-scalemin)/(scalemax-scalemin)
    cv2.imwrite(savedir + 'label.jpg', (inputimg*255).astype(np.uint8))
    inputpic = (inputimg * (1 - mask) * 255).astype(np.uint8)
    cv2.imwrite(savedir + 'input.jpg', inputpic)
    inputimg = inputimg * (1 - mask) + mask * 0.5  # or mask*np.mean(data) if use the mean of data to fill the hole
    inputimg = inputimg.reshape((1, 1, h, w))
    inputimg = torch.tensor(inputimg)
    maskinput = torch.tensor(mask.reshape((1, 1, h, w)))
    # model
    outputimg = model.test_onepic(inputimg, maskinput)

    outputimg = outputimg[0][0]
    recon = (inputimg.numpy().squeeze())*(1-mask)+mask*outputimg
    reconimg = (abs(recon) * 255).astype(np.uint8)
    recon = recon*(scalemax-scalemin)+scalemin
    # SNR
    s = np.sum(testdata ** 2)
    n = np.sum((testdata - recon) ** 2)
    snr = math.log(s / n, 10)
    snr = np.round(snr * 10,2)
    print("SNR:", snr)
    cv2.imwrite(savedir + 'reconimg_'+str(snr)+'.jpg', reconimg)
    np.save(savedir + 'recon_'+str(snr)+'.npy', recon)


if __name__ == '__main__':
    opt._parse()
    ###############post-stack data test###############
    # testpath = "./data/test/seismic2.npy"
    # maskpath = "./data/test/mask5120.5.npy"
    ###############pre-stack data test################
    testpath = "./data/test/data1mobil.npy"
    maskpath = "./data/test/maskmobil_10.3.npy"
    ##################################################
    savedir = './result/testF/'
    data = np.load(testpath).astype(np.float32)
    mask = np.load(maskpath).astype(np.float32)

    trainer = Ftrainer(opt, image_size=opt.image_size)
    trainer.load_F(opt.load_F)
    print('model construct completed')

    test_model(data, mask, trainer, savedir)


