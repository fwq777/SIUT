import cv2
import os
import numpy as np
from torch.utils import data as data_
from tqdm import tqdm
from utils.config import opt
from data.datasetF import Dataset as Datatrain
from data.datawmask import Dataset as Dataval
from utils.vis_tool import Visualizer
from model.DeviceAdapt_Net import Ftrainer

SAVEROOT='./result/result0628/'

def vis_pic(image, snr, vis_tool):
    image = image[0]
    image = image*255
    image = image.astype(np.uint8).transpose((1, 2, 0)).copy()
    text = "snr:"+str(snr)
    font = cv2.FONT_ITALIC
    cv2.putText(image, text, (7, 25), font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
    vis_tool.img("predict", image.transpose((2, 0, 1)))


def test_model(dataloader, model, epoch, ifsave=False, test_num=100000, name='val/'):
    SNR = 0.0
    num = 0
    model.eval()
    dir = SAVEROOT+name+str(epoch)+'/'
    if not os.path.exists(dir) and ifsave:
        os.makedirs(dir)
    for ii, (img, oriimg, mask) in enumerate(dataloader):
        snr, outputimg, seg1, seg2 = model(img, oriimg, mask, vis=True)
        SNR += snr
        num += 1
        if ifsave:
            # for i in range(len(outputimg)):
            i = 0
            mask = mask[i][0].numpy().astype(np.uint8)
            rmask = np.ones_like(mask)
            rmask = rmask - mask
            rmask = rmask.astype(np.uint8)
            img = img[i][0].numpy()
            img = img * 255
            outimg = outputimg[i][0]
            seg1 = seg1[i]*255
            seg1 = seg1.transpose((1, 2, 0))
            seg1 = seg1.astype(np.uint8)
            seg2 = seg2[i]*255
            seg2 = seg2.transpose((1, 2, 0))
            seg2 = seg2.astype(np.uint8)
            outimg = outimg * 255
            oriimg = oriimg[i][0].numpy()
            oriimg = oriimg * 255
            img = img.astype(np.uint8)*rmask
            outimg = outimg.astype(np.uint8)
            oriimg = oriimg.astype(np.uint8)
            reconimg = img * rmask + outimg * mask
            reconimg = reconimg.astype(np.uint8)
            cv2.imwrite(dir + 'out' + str(ii) + '_' + str(i) + '.jpg', outimg)
            cv2.imwrite(dir + 'input' + str(ii) + '_' + str(i) + '.jpg', img)
            cv2.imwrite(dir + 'recon' + str(ii) + '_' + str(i) + '.jpg', reconimg)
            cv2.imwrite(dir + 'ori' + str(ii) + '_' + str(i) + '.jpg', oriimg)
            cv2.imwrite(dir + 'predictseg' + str(ii) + '_' + str(i) + '.jpg', seg1)
            cv2.imwrite(dir + 'oriseg' + str(ii) + '_' + str(i) + '.jpg', seg2)

        if ii > test_num:
            break
    return {"SNR": round(SNR/num, 5)}


def train():
    opt._parse()
    vis_tool = Visualizer(env=opt.env)

    print('load data')
    train_dataset = Datatrain(opt.rootpath, mode="train/")
    val_dataset = Dataval(opt.rootpath, mode="val/")

    trainer = Ftrainer(opt, image_size=opt.image_size)
    if opt.load_G:
        trainer.load_G(opt.load_G)
    print('model G construct completed')

    if opt.load_F:
        trainer.load_F(opt.load_F)
        print('model F construct completed')

    best_map = 0.0
    for epoch in range(opt.epoch):
        trainer.train()
        train_dataloader = data_.DataLoader(train_dataset,
                                            batch_size=opt.train_batch_size,
                                            shuffle=True,
                                            num_workers=opt.num_workers)
        val_dataloader = data_.DataLoader(val_dataset,
                                            batch_size=opt.test_batch_size,
                                            num_workers=opt.num_workers,
                                            shuffle=False)

        # test_model(test_dataloader, trainer, epoch, ifsave=True, test_num=opt.test_num)
        for ii, (img, oriimg, mask) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            loss, loss1, loss2 = trainer.train_onebatch(img, oriimg, mask)
            if ii % 20 == 0:
                trainer.eval()
                vis_tool.plot("totalloss", loss.detach().cpu().numpy())
                vis_tool.plot("loss_r", loss1.detach().cpu().numpy())
                vis_tool.plot("loss_t", loss2.detach().cpu().numpy())
                snr, output, edg, edg2 = trainer(img[0:2, :, :, :], oriimg[0:2, :, :, :], mask[0:2, :, :, :], vis=True)
                vis_tool.plot("snr_train", snr)
                input = img[0][0].numpy()
                input = (input * 255).astype(np.uint8)
                vis_tool.img("input", input)
                label = oriimg[0][0].numpy()
                label = (label * 255).astype(np.uint8)
                vis_tool.img("label", label)
                snr = round(snr, 2)
                vis_pic(output, snr, vis_tool)
                vis_tool.img("predict_segm", edg[0])
                vis_tool.img("ori_segm", edg2[0])
                trainer.train()

        ifsave=False
        if (epoch+1)%10 == 0:
            ifsave=True
        eval_result = test_model(val_dataloader, trainer, epoch, ifsave=ifsave, test_num=opt.test_num)
        print('eval_loss: ', eval_result)

        vis_tool.plot("SNR_val", eval_result["SNR"])
        if epoch > 100 and eval_result["SNR"]>best_map:
            best_map = eval_result["SNR"]
            best_path = trainer.save_F(best_map=best_map)
            print("save to %s !" % best_path)
if __name__ == '__main__':
    train()

