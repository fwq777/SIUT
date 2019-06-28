import math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import time
from model.Unet import UNet
from model.Unet2 import UNet2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class Gtrainer(nn.Module):
    def __init__(self, opt, image_size=128):
        super(Gtrainer, self).__init__()
        self.image_size = image_size
        self.netG = nn.DataParallel(UNet2(1, 3)).to(device)
        self.opt = opt
        self.losscross = nn.CrossEntropyLoss()
        self.optimizer_G = optim.Adam(self.netG.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.lr_scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=opt.lr_step, gamma=opt.lr_decay)

    def train_onebatch(self, image, orisegm):
        self.lr_scheduler_G.step()
        orisegm = orisegm.to(device)
        input = image.to(device)
        output_img = self.netG(input)
        label = orisegm
        loss = self.losscross(output_img, label)
        self.optimizer_G.zero_grad()
        loss.backward()
        self.optimizer_G.step()
        return loss, output_img

    def test_onepic(self, image):
        output = self.netG(image.to(device))
        output = output.detach().cpu().numpy()
        edg = np.ones_like(output)
        output = np.argmax(output, axis=1)
        edg[:, 0, :, :] = edg[:, 0, :, :] * (output == 0)
        edg[:, 1, :, :] = edg[:, 1, :, :] * (output == 1)
        edg[:, 2, :, :] = edg[:, 2, :, :] * (output == 2)
        return edg

    def forward(self, image, orisegm, vis=False):
        if self.training:
            return self.train_onebatch(image, orisegm)

        else:
            output = self.netG(image.to(device))
            loss = self.losscross(output, orisegm.to(device))
            output = output.detach().cpu().numpy()
            edg = np.ones_like(output)
            output = np.argmax(output, axis=1)
            edg[:, 0, :, :] = edg[:, 0, :, :] * (output == 0)
            edg[:, 1, :, :] = edg[:, 1, :, :] * (output == 1)
            edg[:, 2, :, :] = edg[:, 2, :, :] * (output == 2)
            return edg, loss.detach().cpu().numpy()

    def save_G(self, save_path=None, **kwargs):
        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/netG' + '_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_
            save_path = save_path + ".pth"
        torch.save(self.netG.state_dict(), save_path)
        return save_path


    def load_G(self, save_path):
        state_dict = torch.load(save_path)
        self.netG.load_state_dict(state_dict)
        return self


class Ftrainer(nn.Module):
    def __init__(self, opt, image_size=64):
        super(Ftrainer, self).__init__()
        self.image_size = image_size
        self.netF = nn.DataParallel(UNet(2, 1)).to(device)
        self.netG = nn.DataParallel(UNet2(1, 3)).to(device)
        self.opt = opt
        # self.Gan_loss = GANLoss().cuda()
        # self.crossloss = nn.CrossEntropyLoss()
        if opt.mode=='mode1':
            self.loss_r = nn.L1Loss()
            self.loss_t = nn.CrossEntropyLoss()
            self.lambda_r = opt.LAMBDA_DICT["L1"]
            self.lambda_t = opt.LAMBDA_DICT["Texure"]
        else:
            self.loss_r = nn.MSELoss()
            self.loss_t = nn.MSELoss()
            self.lambda_r = opt.LAMBDA_DICT["Recon"]
            self.lambda_t = opt.LAMBDA_DICT["Texure"]
        self.optimizer_F = optim.Adam(self.netF.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.lr_scheduler_F = optim.lr_scheduler.StepLR(self.optimizer_F, step_size=opt.lr_step, gamma=opt.lr_decay)

    def train_onebatch(self, image, oriimg, mask):
        self.lr_scheduler_F.step()
        input = image.to(device)
        mask = mask.to(device)
        label = oriimg.to(device)
        input = torch.cat([input, mask], 1)
        output_img = self.netF(input)
        loss_r = self.loss_r(output_img, label)
        inseg = self.netG(output_img)
        segori = self.netG(label)
        segori = torch.argmax(segori, dim=1)
        loss_t = self.loss_t(inseg, segori)
        loss = loss_r*self.lambda_r+loss_t*self.lambda_t
        self.optimizer_F.zero_grad()
        loss.backward()
        self.optimizer_F.step()
        return loss, loss_r, loss_t

    def test_onepic(self, image, mask):
        output = self.netF(torch.cat([image.to(device), mask.to(device)], 1))
        output = output.detach().cpu().numpy()
        return output

    def forward(self, image, oriimg, mask, vis=False):
        if self.training:
            return self.train_onebatch(image, oriimg, mask)

        else:
            output = self.netF(torch.cat([image.to(device), mask.to(device)], 1))  # torch.cat([image.cuda(), mask.cuda()], 1)
            seg = self.netG(output).detach().cpu().numpy()
            edg = np.ones_like(seg)
            seg = np.argmax(seg, axis=1)
            edg[:, 0, :, :] = edg[:, 0, :, :] * (seg == 0)
            edg[:, 1, :, :] = edg[:, 1, :, :] * (seg == 1)
            edg[:, 2, :, :] = edg[:, 2, :, :] * (seg == 2)
            oriedg = self.netG(oriimg.cuda()).detach().cpu().numpy()
            edg2 = np.ones_like(oriedg)
            oriedg = np.argmax(oriedg, axis=1)
            edg2[:, 0, :, :] = edg2[:, 0, :, :] * (oriedg == 0)
            edg2[:, 1, :, :] = edg2[:, 1, :, :] * (oriedg == 1)
            edg2[:, 2, :, :] = edg2[:, 2, :, :] * (oriedg == 2)
            oriimg = oriimg.numpy()
            mask = mask.numpy()
            output = output.detach().cpu().numpy()
            output1 = output*mask
            label = oriimg*mask
            s=np.sum(oriimg**2)
            n=np.sum((label-output1)*(label-output1))
            snr = math.log(s/n, 10)
            snr = snr*10
            if vis:
                return snr, output, edg, edg2
            else:
                return snr

    def save_F(self, save_path=None, **kwargs):
        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints2/netF' + '_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_
            save_path = save_path + ".pth"
        torch.save(self.netF.state_dict(), save_path)
        return save_path


    def load_F(self, save_path):
        state_dict = torch.load(save_path)
        self.netF.load_state_dict(state_dict)
        return self

    def load_G(self, loadpath):
        state_dict = torch.load(loadpath)
        self.netG.load_state_dict(state_dict)
        return self