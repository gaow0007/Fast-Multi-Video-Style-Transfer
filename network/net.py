import torch.nn as nn
from torchvision.utils import save_image
from torch.autograd import Variable
import torch
from TVLoss import TVLoss

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

def vgg_denorm(var):
    dtype = torch.cuda.FloatTensor
    mean = Variable(torch.zeros(var.size()).type(dtype))
    std = Variable(torch.zeros(var.size()).type(dtype))
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    normed = var.mul(std).add(mean)
    return normed

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


class Net(nn.Module):
    def __init__(self, styler, vgg16):
        super(Net, self).__init__()
        self.styler = styler
        self.vgg = vgg16
        if self.vgg is not None:
            for param in self.vgg.parameters():
                param.requires_grad = False
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss(TVLoss_weight=1)

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (target.requires_grad is False)
        input_gram = gram_matrix(input)
        target_gram = gram_matrix(target)
        return self.mse_loss(input_gram, target_gram)

    def forward(self, content, style, bank, alpha=1.0):
        content = vgg_norm(content)
        style = vgg_norm(style)
        output = self.styler(content, bank)
        content_feat = self.vgg(Variable(content.data, requires_grad=False))[2]
        style_feats = self.vgg(style)
        output_feats = self.vgg(vgg_norm(output))
        loss_c = self.calc_content_loss(output_feats[2], Variable(content_feat.data, requires_grad=False))
        loss_s = 0
        for i in range(4):
            loss_s += self.calc_style_loss(output_feats[i], style_feats[i])
        loss_t = self.tv_loss(output)
        return loss_c, loss_s, loss_t

    def evaluate(self, content, bank):
        with torch.no_grad():
            output = self.styler(vgg_norm(content), bank)
        return output
