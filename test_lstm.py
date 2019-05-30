import argparse
import os
import copy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from network import net
from network import styler2
from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image

import time
import logging
from utils.log_helper import init_log
from torch.autograd import Variable
import mmcv
init_log('global', logging.INFO)
logger = logging.getLogger('global')


cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--style_weight', type=float, default=0.0)
parser.add_argument('--rec_weight', type=float, default=1.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--temporal_weight', type=float, default=1.0)
parser.add_argument('--total_weight', type=float, default=2e-8)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--print_freq', type=int, default=20)

def train_transform():
    transform_list = [
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def style_transform():
    transform_list = [
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)




class ContentDataset(data.Dataset):
    def __init__(self, root, transform):
        super(ContentDataset, self).__init__()
        self.root = root
        with open('imgout.txt') as f:
            lines = f.readlines()
        self.paths = []
        for path in os.listdir(self.root):
            mpath = os.path.join(self.root, path) + '\n'
            if mpath in lines:
                print(mpath, flush=True)
                continue
            self.paths.append(path)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'ContentDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def styleInput():
    imgs = []
    style_tf = style_transform()
    for i in range(10):
        path = '{}.jpg'.format(i)
        img = Image.open(os.path.join('./style', path)).convert('RGB')
        img = style_tf(img).unsqueeze(0)
        img_arr = []
        for j in range(args.batch_size):
            img_arr.append(img)
        img = torch.cat(img_arr, dim=0)
        print(img.shape, flush=True)
        imgs.append(img)
    return imgs

def vgg_norm(var):
    dtype = torch.cuda.FloatTensor
    mean = Variable(torch.zeros(var.size()).type(dtype))
    std = Variable(torch.zeros(var.size()).type(dtype))
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    normed = var.sub(mean).div(std)
    return normed

if __name__ == '__main__':
    args = parser.parse_args()
    from network.vgg16 import Vgg16
    styler = styler2.ReCoNet()
    styler.eval()
    # styler.load_state_dict(torch.load('exp/set6/checkpoint/decoder_iter_10.pth.tar'), strict=True)
    # styler.load_state_dict(torch.load('exp/set8/checkpoint/decoder_iter_50.pth.tar'))
    # styler.load_state_dict(torch.load('experiments-c32//model_iter_320000.pth.tar')['state_dict'], strict=False)
    # styler.load_state_dict(torch.load('experiments/model_iter_600000.pth.tar')['state_dict'], strict=True)
    styler.load_state_dict(torch.load('exp/set8/checkpoint/decoder_iter_85.pth.tar')['state_dict'], strict=True)

    styler = styler.cuda()



    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    content_tf = train_transform()
    style_tf = train_transform()

    print('loading dataset done', flush=True)

    # style_bank = styleInput()
    from utils.utils import repackage_hidden
    print(styler, flush=True)
    avg = []
    for bank in range(120):
        prev_state1 = None
        prev_state2 = None
        for i in range(1, 10):
            path = '%05d.jpg'%(i)
            cimg = Image.open(os.path.join('/home/gaowei/IJCAI/videvo/videvo/test/WaterFall2/', path)).convert('RGB')

            cimg = content_tf(cimg).unsqueeze(0).cuda()
            cimg = vgg_norm(cimg)
            with torch.no_grad():
                out, prev_state1, prev_state2 = styler(cimg, prev_state1, prev_state2, bank)


            prev_state1 = repackage_hidden(prev_state1)
            prev_state2 = repackage_hidden(prev_state2)
            save_image(out, 'output/%06d.jpg'%(i-1 + bank * 49))
    # mmcv.frames2video('output', 'mst_cat_flow.avi', fps=6)


