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
from network import styler
from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image

import time
import logging
from utils.log_helper import init_log
from torch.autograd import Variable

torch.cuda.set_device(0)
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
    for i in range(21):
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

if __name__ == '__main__':
    args = parser.parse_args()
    from network.vgg16 import Vgg16
    styler = styler.ReCoNet()
    styler.eval()
    # styler.load_state_dict(torch.load('experiments-flow/decoder_iter_5.pth.tar'), strict=True)


    vgg = Vgg16()
    vgg.eval()
    network = net.Net(styler, vgg)
    network = network.cuda()

    if args.parallel:
        network = nn.DataParallel(network)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    content_tf = train_transform()
    style_tf = train_transform()

    print('loading dataset done', flush=True)

    # style_bank = styleInput()

    outputdir = 'sinteloutput/'
    contentdir = '/home/gaowei/IJCAI/MPI/test/clean/'
    # contentdir = '/mnt/lustre/share/gaowei_exp/MPI/Sintel/test/'
    videvo = [
            'bamboo_3',
            'cave_3',
            'market_1',
            'mountain_2',
            'temple_1',
            'wall']

    for bank in range(21):
        # if bank not in [5]:
        if bank not in [0, 5, 8, 10, 17]:
            continue
        print('bank ============================================= ', bank)
        styler.load_state_dict(torch.load('experiments-flow-s{}/decoder_iter_best.pth.tar'.format(bank)), strict=True)
        for clabel in os.listdir(contentdir):
            if clabel not in videvo:
                continue
            print(clabel, bank, flush=True)
            if not os.path.exists('{}/{}'.format(outputdir, clabel)):
                os.mkdir('{}/{}'.format(outputdir, clabel))
            maxlen = min(50, len(os.listdir('{}/{}'.format(contentdir, clabel))))
            print('maaxlen, ', maxlen)
            for i in range(1, maxlen):
                # path = '%05d.jpg'%(i)
                path = 'frame_%04d.png'%(i)
                cimg = Image.open(os.path.join('{}/{}/'.format(contentdir, clabel), path)).convert('RGB')
                cimg = content_tf(cimg).unsqueeze(0).cuda()
                out = network.evaluate(cimg, bank)
                save_image(out, '%s/%s/%06d.jpg'%(outputdir, clabel, (i-1) + bank * 49))

