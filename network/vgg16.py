import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np



class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        print('Preparing pretrained VGG 16 ...')
        model = torchvision.models.vgg16(pretrained=True)
        # model.load_state_dict(torch.load('/home/gaowei/vgg16-397923af.pth'))
        self.vgg_16 = model.features

        self.relu_1_2 = nn.Sequential(*list(self.vgg_16.children())[0:4])
        self.relu_2_2 = nn.Sequential(*list(self.vgg_16.children())[4:9])
        self.relu_3_3 = nn.Sequential(*list(self.vgg_16.children())[9:16])
        self.relu_4_3 = nn.Sequential(*list(self.vgg_16.children())[16:23])

    def forward(self, x):
        out_1_2 = self.relu_1_2(x)
        out_2_2 = self.relu_2_2(out_1_2)
        out_3_3 = self.relu_3_3(out_2_2)
        out_4_3 = self.relu_4_3(out_3_3)

        return [out_1_2, out_2_2, out_3_3, out_4_3]
