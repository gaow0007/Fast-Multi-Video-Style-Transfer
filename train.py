import argparse
import numpy as np
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
from utils.utils import vgg_norm, vgg_denorm

import time
import logging
from utils.log_helper import init_log
from torch.autograd import Variable
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
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--print_freq', type=int, default=20)
parser.add_argument('--bank', type=int, default=0)

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    # transforms.RandomCrop(256),
    return transforms.Compose(transform_list)

def style_transform():
    transform_list = [
        transforms.Resize(size=(384, 384)),
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
    for i in range(120):
        path = '{}.jpg'.format(i)
        img = Image.open(os.path.join('./style', path)).convert('RGB')
        img = style_tf(img).unsqueeze(0)
        img_arr = []
        for j in range(args.batch_size):
            img_arr.append(img)
        img = torch.cat(img_arr, dim=0)
        imgs.append(img)
    return imgs

if __name__ == '__main__':
    args = parser.parse_args()
    from network.vgg16 import Vgg16
    styler = styler.ReCoNet()
    styler.train()

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
    content_dataset = ContentDataset('/home/share/MSCOCO2017/', content_tf)

    content_loader = data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads)

    print('loading dataset done', flush=True)

    style_bank = styleInput()

    optimizer = torch.optim.Adam(network.styler.parameters(), lr=args.lr, weight_decay=5e-4)

    content_iter = iter(content_loader)
    t0 = time.time()

    for i in range(800000):
        bank = np.random.randint(120)
        style_images = style_bank[bank]
        style_images = Variable(style_images.cuda(), requires_grad=False)
        content_images = Variable(next(content_iter).cuda(), requires_grad=True)

        loss_c, loss_s, loss_t = network(content_images, style_images, bank)
        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss = loss_c + loss_s  + loss_t

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t2 = time.time()
        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            torch.save({'state_dict': network.styler.state_dict()}, '{:s}/model_iter_{:d}.pth.tar'.format(args.save_dir, i + 1))
        if (i + 1) % 20 == 0:
            logger.info('Iter: [%d] LR:%f Time: %.3f Loss: %.5f LossContet: %.5f  LossSytle: %.5f LossTV: %.5f' % (i+1, args.lr, t2 - t0, loss.data.cpu().item(), loss_c.data.cpu().item(), loss_s.data.cpu().item(), loss_t.data.cpu().item()))
            t0 = t2


