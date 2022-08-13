import argparse
import os
from os.path import exists, join as join_paths
import torchvision.transforms as transforms
import torch
import numpy as np
from skimage import io, color
from skimage.transform import resize
from utils.utils import labimage2tensor, tensor2img
from models.model import ConGenerator_S2F, ConRefineNet
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_1', type=str, default='pretrained/netG_1_aistd.pth', help='generator_1 checkpoint file')
parser.add_argument('--generator_2', type=str, default='pretrained/netG_2_aistd.pth', help='generator_2 checkpoint file')
parser.add_argument('--savepath', type=str, default='results/aistd/', help='save path')
parser.add_argument('--dataset', type=str, default='aistd', help='save path')
opt = parser.parse_args()

if opt.dataset == 'aistd':
    ## ISTD dataset
    opt.dataroot_A = 'input/dataset/ISTD/test/test_A'
    opt.im_suf_A = '.png'
    opt.dataroot_B = 'input/BDRAR/test_A_mask_istd_6/'
    opt.im_suf_B = '.png'
elif opt.dataset == 'srd':
    ## SRD dataset
    opt.dataroot_A = 'input/dataset/SRD/test/shadow'
    opt.im_suf_A = '.jpg'
    opt.dataroot_B = 'input/dataset/SRD/test/SRD_testmask_dil_2'
    opt.im_suf_B = '.jpg'
else:
    print("Please check the name of dataset...")
    exit(0)

if torch.cuda.is_available():
    opt.cuda = True
    device = torch.device('cuda:0')
print(opt)

## save dir
if not os.path.exists(opt.savepath):
    os.makedirs(opt.savepath)

###### Definition of variables ######
# Networks
netG_1 = ConGenerator_S2F()
netG_2 = ConRefineNet()

if opt.cuda:
    netG_1.to(device)
    netG_2.to(device)

netG_1.load_state_dict(torch.load(opt.generator_1))
netG_2.load_state_dict(torch.load(opt.generator_2))
netG_1.eval()
netG_2.eval()

gt_list = [os.path.splitext(f)[0] for f in os.listdir(opt.dataroot_A) if f.endswith(opt.im_suf_A)]

###### evaluation ######
for idx, img_name in enumerate(gt_list):
    # Set model input
    with torch.no_grad():
        labimage = color.rgb2lab(io.imread(os.path.join(opt.dataroot_A, img_name + opt.im_suf_A)))
        h = labimage.shape[0] - labimage.shape[0]%4
        w = labimage.shape[1] - labimage.shape[1]%4
        labimage = labimage2tensor(labimage, h, w).to(device)

        mask = io.imread(os.path.join(opt.dataroot_B, img_name + opt.im_suf_B))
        mask = resize(mask,(h,w))
        mask = torch.from_numpy(mask).float()
        mask = mask.view(h,w,1)
        mask = mask.transpose(0, 1).transpose(0, 2).contiguous()
        mask = mask.unsqueeze(0).to(device)
        zero = torch.zeros_like(mask)
        one = torch.ones_like(mask)
        mask=torch.where(mask > 0.5, one, zero)
        inv_mask = 1.0 - mask

        fake_temp = netG_1(labimage,mask)
        input2 = (labimage * inv_mask + fake_temp * mask)
        fake_temp = netG_2(input2,mask)

        outputimage = tensor2img(fake_temp, h, w)
        save_path = join_paths(opt.savepath+'/%s'% (img_name + opt.im_suf_A))
        io.imsave(save_path,outputimage)

        print('Generated images %04d of %04d' % (idx+1, len(gt_list)))
        