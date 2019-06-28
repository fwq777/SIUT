import cv2
import os
import numpy as np
from torch.utils import data as data_
from tqdm import tqdm
from utils.config import opt
from data.datasetG import Dataset
from utils.vis_tool import Visualizer
from model.DeviceAdapt_Net import Gtrainer


def test_model(dataloader, model, epoch, ifsave=False, test_num=100000):
    LOSS = 0.0
    num = 0
    model.eval()
    dir = './result/resultG0405/'+str(epoch)+'/'
    if not os.path.exists(dir) and ifsave:
        os.makedirs(dir)
    for ii, (img, oriimg) in enumerate(dataloader):
        outputimg, loss = model(img, oriimg, vis=True)
        LOSS += loss
        num += 1
        if ifsave:
            # for i in range(len(outputimg)):
            i = 0
            img = img[i][0].numpy()
            img = img*255
            outimg = outputimg[i]
            outimg = outimg.transpose((1, 2, 0))
            outimg = outimg*255
            img = img.astype(np.uint8)
            outimg = outimg.astype(np.uint8)
            cv2.imwrite(dir + 'out' + str(ii) + '_' + str(i) + '.jpg', outimg)
            cv2.imwrite(dir + 'input' + str(ii) + '_' + str(i) + '.jpg', img)

        if ii > test_num:
            break
    return {"SNR": round(LOSS/num, 5)}


def train():
    opt._parse()
    vis_tool = Visualizer(env=opt.env)
    print("init vis_tool")

    print('load data')
    train_dataset = Dataset(opt.rootpath, mode="train/")
    val_dataset = Dataset(opt.rootpath, mode="val/")

    trainer = Gtrainer(opt, image_size=opt.image_size)
    # if opt.load_G:
    #     trainer.load_G(opt.load_G)
    # print('model construct completed')

    best_map = 0.0
    for epoch in range(opt.epoch):
        trainer.train()
        train_dataloader = data_.DataLoader(train_dataset,
                                            batch_size=opt.train_batch_size,
                                            num_workers=opt.num_workers,
                                            shuffle=True)
        val_dataloader = data_.DataLoader(val_dataset,
                                          batch_size=opt.test_batch_size,
                                          num_workers=opt.num_workers,
                                          shuffle=False)
        # test_model(test_dataloader, trainer, epoch, ifsave=True, test_num=opt.test_num)
        for ii, (img, oriseg) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            trainer.train_onebatch(img, oriseg)
            if ii % 50 == 0:
                trainer.eval()
                outputimg, loss = trainer(img, oriseg, vis=True)
                vis_tool.plot("loss", loss)
                input = img[0][0].numpy()
                input = (input*255).astype(np.uint8)
                vis_tool.img("input", input)
                label = oriseg[0].numpy()
                label = (label*255).astype(np.uint8)
                vis_tool.img("label", label)
                trainer.train()

        ifsave=False
        if (epoch+1)%1 == 0:
            ifsave=True
        eval_result = test_model(val_dataloader, trainer, epoch, ifsave=ifsave, test_num=opt.test_num)
        print('eval_loss: ', eval_result)
        best_map = eval_result["SNR"]
        best_path = trainer.save_G(best_map=best_map)
        print("save to %s !" % best_path)

if __name__ == '__main__':
    train()

