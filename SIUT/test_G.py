import cv2
import os
import numpy as np
from utils.config import opt
from model.DeviceAdapt_Net import Gtrainer
import torch


def test_model(testdata, model, savedir):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    np.save(savedir + 'input.npy', testdata)
    model.eval()
    h, w = testdata.shape
    scalemax = np.max(testdata)
    scalemin = np.min(testdata)
    testdata = (testdata-scalemin)/(scalemax-scalemin)
    inputimg = testdata.reshape((1, 1, h, w))
    inputimg = torch.tensor(inputimg)
    outputimg = model.test_onepic(inputimg)
    outputimg = outputimg[0]
    outputimg = outputimg.transpose((1, 2, 0))

    np.save(savedir + 'output.npy', outputimg)
    outimg = (outputimg*255).astype(np.uint8)
    img = (abs(testdata) * 255).astype(np.uint8)
    cv2.imwrite(savedir + 'out.jpg', outimg)
    cv2.imwrite(savedir + 'input.jpg', img)

if __name__ == '__main__':
    opt._parse()
    testpath = "./data/test/seismic2.npy"
    savedir = './result/testG/'
    data = np.load(testpath).astype(np.float32)
    trainer = Gtrainer(opt, image_size=opt.image_size)
    trainer.load_G(opt.load_G)
    print('model construct completed')

    test_model(data, trainer, savedir)


