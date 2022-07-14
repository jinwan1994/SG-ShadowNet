import random
import time
import datetime
import sys
import torch.nn as nn
from torch.autograd import Variable
import torch
# from visdom import Visdom
import torchvision.transforms as transforms
import numpy as np
from skimage.filters import threshold_otsu
from skimage import io, color
from skimage.transform import resize
from skimage.color import rgb2lab

to_pil = transforms.ToPILImage()
to_gray = transforms.Grayscale(num_output_channels=1)


def labimage2tensor(labimage, h=480, w=640):

    labimage_t = resize(labimage,(h,w,3))
    labimage_t[:,:,0] = np.asarray(labimage_t[:,:,0])/50.0-1.0
    labimage_t[:,:,1:] = 2.0*(np.asarray(labimage_t[:,:,1:])+128.0)/255.0-1.0
    labimage_t = torch.from_numpy(labimage_t).float()
    labimage_t = labimage_t.view(h, w, 3)
    labimage_t = labimage_t.transpose(0, 1).transpose(0, 2).contiguous()
    labimage_t = labimage_t.unsqueeze(0)
    return labimage_t

def tensor2img(tensor, h=480, w=640):

    labimg = tensor.data
    labimg[:,0] = 50.0*(labimg[:,0]+1.0)
    labimg[:,1:] = 255.0*(labimg[:,1:]+1.0)/2.0-128.0
    labimg = labimg.data.squeeze(0).cpu()
    labimg = labimg.transpose(0, 2).transpose(0, 1).contiguous().numpy()
    labimg = resize(labimg,(h,w,3))
    outputimag = color.lab2rgb(labimg)
    return outputimag

def calc_RMSE(real_img, fake_img):
    # convert to LAB color space
    real_lab = rgb2lab(real_img)
    fake_lab = rgb2lab(fake_img)
    return real_lab - fake_lab

class QueueMask_llab():
    def __init__(self, length):
        self.max_length = length
        self.queue = []
        self.queue_L = []

    def insert(self, mask,mask_L):
        if self.queue.__len__() >= self.max_length:
            self.queue.pop(0)
        if self.queue_L.__len__() >= self.max_length:
            self.queue_L.pop(0)

        self.queue.append(mask)
        self.queue_L.append(mask_L)

    def rand_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        assert self.queue_L.__len__() > 0, 'Error! Empty queue!'
        temp=np.random.randint(0, self.queue.__len__())
        return self.queue[temp],self.queue_L[temp]

    def last_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        assert self.queue_L.__len__() > 0, 'Error! Empty queue!'
        return self.queue[self.queue.__len__()-1],self.queue_L[self.queue.__len__()-1]

class QueueMask():
    def __init__(self, length):
        self.max_length = length
        self.queue = []

    def insert(self, mask):
        if self.queue.__len__() >= self.max_length:
            self.queue.pop(0)

        self.queue.append(mask)

    def rand_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        return self.queue[np.random.randint(0, self.queue.__len__())]

    def last_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        return self.queue[self.queue.__len__()-1]

def mask_generator(shadow, shadow_free):
    im_f = to_gray(to_pil(((shadow_free.data.squeeze(0) + 1.0) * 0.5).cpu()))
    im_s = to_gray(to_pil(((shadow.data.squeeze(0) + 1.0) * 0.5).cpu()))

    diff = (np.asarray(im_f, dtype='float32')- np.asarray(im_s, dtype='float32')) # difference between shadow image and shadow_free image
    L = threshold_otsu(diff)
    mask = torch.tensor((np.float32(diff >= L)-0.5)/0.5).unsqueeze(0).unsqueeze(0).cuda() #-1.0:non-shadow, 1.0:shadow
    mask.requires_grad = False

    return mask
    
def cyclemask_generator(shadow, shadow_free):
    im_f = to_gray(to_pil(((shadow_free.data.squeeze(0) + 1.0) * 0.5).cpu()))
    im_s = to_gray(to_pil(((shadow.data.squeeze(0) + 1.0) * 0.5).cpu()))

    diff = (np.asarray(im_f, dtype='float32')- np.asarray(im_s, dtype='float32')) # difference between shadow image and shadow_free image
    L = threshold_otsu(diff)
    L=L*0.1
    mask = torch.tensor(np.float32(diff <= L)).unsqueeze(0).unsqueeze(0).cuda() #0:shadow, 1.0:non-shadow
    mask.requires_grad=False
    return mask


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class cyclemaskloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,fake_B,real_A,mask):
        mask=(1.0-mask)/2.0
        mask=mask.repeat(1,3,1,1)
        mask.requires_grad=False
        return torch.mean(torch.pow((torch.mul(fake_B,mask)-torch.mul(real_A,mask)), 2))


