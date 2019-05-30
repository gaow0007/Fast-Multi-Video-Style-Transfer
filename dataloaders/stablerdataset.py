import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from numpy import random

def test_transform():
    transform_list = [transforms.Resize(size=(256, 256)),
            transforms.ToTensor()]
    transform = transforms.Compose(transform_list)
    return transform

def style_transform():
    transform_list = [
            transforms.ToTensor()]
    transform = transforms.Compose(transform_list)
    return transform

def style_small_transform():
    # to prevent exceeding gpu memory
    transform_list = [transforms.Resize(size=(256, 256)),
            transforms.ToTensor()]
    transform = transforms.Compose(transform_list)
    return transform

class VideoDataset(Dataset):

    def __init__(self, dataset='Sintel', split='train', clip_len=16, bank=0, preprocess=False):
        self.dataset = dataset
        self.bank = bank

        if self.dataset == 'Sintel':
            self.root_dir = '/home/gaowei/MPI/training'
        elif dataset == 'VideoNet':
            self.root_dir = '/home/gaowei/videvo/videvo/train'

        folder = self.root_dir

        self.style_dir = './style/'
        self.transform = test_transform()
        self.styletf = style_small_transform()
        style = Image.open(os.path.join(self.style_dir, '%d.jpg'%(self.bank))).convert('RGB')
        if style.size[1] >= 1000:
            self.styletf = style_small_transform()

        self.clip_len = clip_len
        self.fnames = []

        if self.dataset == 'Sintel':
            for label in sorted(os.listdir(folder)):
                dirlen = len(os.listdir(os.path.join(folder, label)))
                for fname in range(dirlen-2):
                    self.fnames.append((os.path.join(folder, label), label, fname+1))
        elif self.dataset == 'VideoNet':
            for label in sorted(os.listdir(folder)):
                dirlen = len(os.listdir(os.path.join(folder, label)))
                for fname in range(0, dirlen-2, 10):
                    self.fnames.append((os.path.join(folder, label), label, fname))


    def __len__(self):
        return len(self.fnames)


    def __getitem__(self, index):
        dirname, imglabel, imgname = self.fnames[index]

        if self.dataset == 'Sintel':
            img1, img2, style = self.load_frames_sintel(dirname, imglabel, imgname)
            return img1, img2, style
        else:
            img1, img2, style = self.load_frames(dirname, imglabel, imgname)
            return img1, img2, style

    def load_frames(self, dirname, imglabel, imgname):
        img1 = Image.open(os.path.join(dirname, '%05d.jpg'%(imgname))).convert('RGB')
        img1 = self.transform(img1)
        img2 = Image.open(os.path.join(dirname, '%05d.jpg'%(imgname+random.randint(3)+1))).convert('RGB')
        img2 = self.transform(img2)

        styleidx = self.bank
        style = Image.open(os.path.join(self.style_dir, '%d.jpg'%(styleidx))).convert('RGB')
        style = self.styletf(style)

        return img1, img2, style


    def load_frames_sintel(self, dirname, imglabel, imgname):
        img1 = Image.open(os.path.join(dirname, 'frame_%04d.png'%(imgname))).convert('RGB')
        img1 = self.transform(img1)
        img2 = Image.open(os.path.join(dirname, 'frame_%04d.png'%(imgname+random.randint(7)+1))).convert('RGB')
        img2 = self.transform(img2)

        styleidx = self.bank
        style = Image.open(os.path.join(self.style_dir, '%d.jpg'%(styleidx))).convert('RGB')
        style = self.styletf(style)

        return img1, img2, style


