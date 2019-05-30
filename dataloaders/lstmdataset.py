import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from numpy import random
import utils.utils as utils
import torch

def transform():
    stransform = transforms.Compose([transforms.Resize(size=(256, 256)),
                                    transforms.ToTensor()])
    return stransform


class VideoDataset(Dataset):

    def __init__(self, dataset='Sintel', split='train', clip_len=16, preprocess=False):
        self.dataset = dataset

        if self.dataset == 'Sintel':
            self.root_dir = '/home/gaowei/MPI/training'
        elif dataset == 'VideoNet':
            self.root_dir = '/home/gaowei/videvo/videvo/train'

        folder = self.root_dir

        self.transform = transform()
        self.styletf = transform()
        self.clip_len = clip_len
        self.style_dir = './style/'
        self.fnames = []

        if self.dataset == 'Sintel':
            for label in sorted(os.listdir(folder)):
                dirlen = len(os.listdir(os.path.join(folder, label)))
                for fname in range(dirlen-10):
                    self.fnames.append((os.path.join(folder, label), label, fname+1))
        elif self.dataset == 'VideoNet':
            for label in sorted(os.listdir(folder)):
                dirlen = len(os.listdir(os.path.join(folder, label)))
                for fname in range(0, dirlen-10, 10):
                    self.fnames.append((os.path.join(folder, label), label, fname))

    def __len__(self):
        return len(self.fnames)


    def __getitem__(self, index):
        # loading and preprocessing.
        dirname, imglabel, imgname = self.fnames[index]
        if self.dataset == 'Sintel':
            imgs, style = self.load_frames_sintel(dirname, imglabel, imgname)
            return imgs, style
        elif self.dataset == 'VideoNet':
            imgs, style = self.load_frames_videonet(dirname, imglabel, imgname)
            return imgs, style
        else:
            raise NotImplementedError


    def load_frames_videonet(self, dirname, imglabel, imgname):
        imgs = []
        for i in range(10):
            img = Image.open(os.path.join(dirname, '%05d.jpg'%(imgname+i))).convert('RGB')
            img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)
        style = Image.open(os.path.join(self.style_dir, '%d.jpg'%(0)))
        style = self.styletf(style)
        return imgs, style


    def load_frames_sintel(self, dirname, imglabel, imgname):
        imgs = []
        for i in range(10):
            img = Image.open(os.path.join(dirname, 'frame_%04d.png'%(imgname+i))).convert('RGB')
            img = self.transform(img).unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)
        style = Image.open(os.path.join(self.style_dir, '%d.jpg'%(0)))
        style = self.styletf(style)
        return imgs, style
