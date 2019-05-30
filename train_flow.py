import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloaders.stablerdataset import VideoDataset
from flow.models.FlowNetS import FlowNetS
from network import stablernet
from network import styler

import time
import logging
from utils.log_helper import init_log

torch.backends.cudnn.enabled = False
init_log('global', logging.INFO)
logger = logging.getLogger('global')


cudnn.benchmark = False
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def dataloader(dataset, batch_size, bank):
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, bank=bank), batch_size=batch_size, shuffle=True, num_workers=4)
    return train_dataloader

parser = argparse.ArgumentParser()

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--style_weight', type=float, default=1.0)
parser.add_argument('--rec_weight', type=float, default=1.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--temporal_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--print_freq', type=int, default=20)
parser.add_argument('--dataset', type=str, default='Penn')
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--bank', type=int, default=0)

args = parser.parse_args()

device = torch.device('cuda')

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

flownet = FlowNetS(batchNorm=False)
flownet.load_state_dict(torch.load('./flow/pretrained/flownets.pth.tar')['state_dict'], strict=False)

mstyler = styler.ReCoNet()
print(mstyler, flush=True)
mstyler.load_state_dict(torch.load('experiments-s{}/model_best.pth.tar'.format(args.bank))['state_dict'], strict=True)
# mstyler.load_state_dict(torch.load('experiments-flow-s{}/decoder_iter_best.pth.tar'.format(args.bank)), strict=True)
mstyler.train()


from network.vgg16 import Vgg16
vgg = Vgg16()
# vgg.load_state_dict(torch.load('vgg_models/vgg16.weight'), strict=False)
vgg.eval()


network = stablernet.VideoNet(mstyler, vgg, flownet)

if args.parallel:
    network = nn.DataParallel(network)
network.to(device)


def train(model, train_loader, epoch, save_dir, optimizer):
    datalen = len(train_loader)
    t0 = time.time()
    for i, inputs in enumerate(train_loader):
        content1, content2, style = inputs
        loss_c, loss_s, loss_t = model(content1.to(device), content2.to(device), style.to(device), args.bank, (i) % 40 == 0, 0.6)
        loss_c = loss_c * args.content_weight
        loss_s = loss_s * args.style_weight
        loss_t = loss_t * args.temporal_weight
        loss = loss_t + loss_s + loss_c

        if args.parallel:
            loss = loss.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t2 = time.time()

        if (i + 1) % args.print_freq == 0:
            logger.info('Epoch [%d] Iter: [%d/%d] LR:%f Time: %.3f Loss: %.5f LossContent: %.5f  LossStyle: %.5f LossTemporal: %.5f' %
                    (epoch, i+1, datalen, args.lr, t2 - t0, loss.data.cpu().item(), loss_c.data.cpu().item(), loss_s.data.cpu().item(), loss_t.data.cpu().item()))
            t0 = t2


if __name__ == '__main__':
    network.to(device)
    param_groups = []
    param_groups.append({'params' : network.styler.parameters()})
    optimizer = torch.optim.Adam(param_groups, lr=args.lr)
    train_loader = dataloader(args.dataset, args.batch_size, args.bank)
    network.flownet.eval()
    network.styler.train()


    for epoch in range(args.epoch):
        train(model=network, train_loader=train_loader, epoch=epoch, save_dir=args.save_dir, optimizer=optimizer)
        if (epoch+1) % 1 == 0:
            state_dict = network.styler.state_dict()
            torch.save(state_dict,
                       '{:s}/decoder_iter_best.pth.tar'.format(args.save_dir, epoch + 1))


