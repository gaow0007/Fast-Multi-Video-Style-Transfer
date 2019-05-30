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
import dataloaders
from torch.utils.data import DataLoader
from dataloaders.lstmdataset import VideoDataset
from flow.models.FlowNetS import FlowNetS
from network import lstmnet
from network import styler2

import time
import logging
from utils.log_helper import init_log
from utils.utils import repackage_hidden
init_log('global', logging.INFO)
logger = logging.getLogger('global')


cudnn.benchmark = False
torch.backends.cudnn.enabled = False
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def dataloader(dataset, batch_size):
    train_dataloader = DataLoader(VideoDataset(dataset=dataset), batch_size=batch_size, shuffle=True, num_workers=4)
    return train_dataloader

def style_transform():
    transform_list = [
        transforms.Resize(size=(384, 384)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=1.0)
parser.add_argument('--rec_weight', type=float, default=1.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--short_weight', type=float, default=1.0)
parser.add_argument('--long_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--print_freq', type=int, default=20)
parser.add_argument('--dataset', type=str, default='Penn')
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--epoch', type=int, default=100)

args = parser.parse_args()

device = torch.device('cuda')

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

flownet = FlowNetS(batchNorm=False)
flownet.load_state_dict(torch.load('./flow/pretrained/flownets.pth.tar')['state_dict'], strict=False)

mstyler = styler2.ReCoNet()
mstyler.load_state_dict(torch.load('experiments/model_iter_600000.pth.tar')['state_dict'], strict=False)
mstyler.train()


from network.vgg16 import Vgg16
vgg = Vgg16()
vgg.eval()


network = lstmnet.VideoNet(mstyler, vgg, flownet)
print(network, flush=True)

if args.parallel:
    network = nn.DataParallel(network)
network.to(device)


def train(model, train_loader, epoch, save_dir, optimizer, style_bank):
    from numpy import random
    datalen = len(train_loader)
    t0 = time.time()
    for i, inputs in enumerate(train_loader):
        if i == datalen -1:
            break
        bank = random.randint(120)
        style = style_bank[bank]
        prev_state1 = None
        prev_state2 = None
        contents = inputs
        frame_i = []
        frame_o = []
        for t in range(10):
            frame_i.append(contents[:, t, :, :, :])
        loss = 0
        for t1 in range(9):
            loss_c, loss_s, loss_t, out, prev_state1, prev_state2 = model(frame_i[t1].to(device), frame_i[t1+1].to(device), style.to(device), prev_state1, prev_state2, bank)
            prev_state1 = repackage_hidden(prev_state1)
            prev_state2 = repackage_hidden(prev_state2)

            frame_o.append(out.detach())
            loss_c = loss_c * args.content_weight
            loss_s = loss_s * args.style_weight
            loss_t = loss_t * args.short_weight
            loss = (loss_c + loss_s + loss_t)
            for t2 in range(1, t1-4):
                loss_t = model.temporal_loss(frame_i[t2].to(device), frame_i[t1+1].to(device), frame_o[t2-1].to(device), out.to(device))
                loss_t = loss_t * args.long_weight / (t1-5)
                loss += loss_t
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t2 = time.time()

        # if (i + 1) % args.print_freq == 0:
        if True:
            logger.info('Epoch [%d] Iter: [%d/%d] LR:%f Time: %.3f Loss: %.5f LossContent: %.5f  LossStyle: %.5f LossTemporal: %.5f' %
                    (epoch, i+1, datalen, args.lr, t2 - t0, loss.data.cpu().item(), loss_c.data.cpu().item(), loss_s.data.cpu().item(), loss_t.data.cpu().item()))
            t0 = t2


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
    network.to(device)
    param_groups = []
    param_groups.append({'params' : network.styler.parameters()})
    optimizer = torch.optim.Adam(param_groups, lr=args.lr)
    train_loader = dataloader(args.dataset, args.batch_size)
    network.flownet.eval()
    network.styler.train()
    style_bank = styleInput()
    network.set_train()


    for epoch in range(args.epoch):
        train(model=network, train_loader=train_loader, epoch=epoch, save_dir=args.save_dir, optimizer=optimizer, style_bank=style_bank)
        if (epoch+1) % 5 == 0:
            state_dict = network.styler.state_dict()
            torch.save(state_dict,
                       '{:s}/decoder_iter_{:d}.pth.tar'.format(args.save_dir, epoch + 1))


