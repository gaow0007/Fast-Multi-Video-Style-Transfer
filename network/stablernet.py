import torch.nn as nn

import torch
from torch.autograd import Variable
from torchvision.utils import save_image


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

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


class VideoNet(nn.Module):
    def __init__(self, styler, vgg, flow):
        super(VideoNet, self).__init__()
        self.styler = styler
        self.vgg = vgg
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.flownet = flow
        self.flownet.eval()

        for param in self.vgg.parameters():
            param.requires_grad = False
        for parm in self.flownet.parameters():
            param.requires_grad = False


    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size()[:2] == target.size()[:2])
        assert (target.requires_grad is False)
        input_gram = gram_matrix(input)
        target_gram = gram_matrix(target)
        return self.mse_loss(input_gram, target_gram)

    def concat(self, content1, content2):
        content = torch.cat([content1, content2], dim=1)
        return content

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        if flo.size(2) < x.size(2):
            scale_factor = x.size(2) // flo.size(2)
            flo = torch.nn.functional.upsample(flo, size=x.size()[-2:], mode='bilinear')  * scale_factor
        elif flo.size(2) > x.size(2):
            scale_factor = flo.size(2) // x.size(2)
            flo = torch.nn.functional.avg_pool2d(flo, scale_factor)  / scale_factor

        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)
        output = nn.functional.grid_sample(x, vgrid)

        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        return output*mask

    def mask_occlusion(self, img1, img2_, alpha=50.0):
        return torch.exp( -alpha * torch.sum(img1 - img2_, dim=1).pow(2)  ).unsqueeze(1)

    def calc_temporal_loss(self, img1, img2, flow, mask):
        return self.l1_loss(mask * self.warp(img2, flow),  Variable(mask.data * img1.data, requires_grad=False))

    def forward(self, content1, content2, style, bank, debug=False, alpha=1.0):
        assert 0 <= alpha <= 1
        with torch.no_grad():
            contents = self.concat(content1, content2)
            flowout = self.flownet(contents)
            mask = self.mask_occlusion(content1, self.warp(content2, flowout))
            style_feats = self.vgg(vgg_norm(style))
            content_feat = self.vgg(vgg_norm(Variable(content2.data, requires_grad=False)))[2]
            g_t1 = self.styler(vgg_norm(content1), bank)

        g_t2 = self.styler(vgg_norm(content2), bank)

        output_feats = self.vgg(vgg_norm(g_t2))

        loss_c = self.calc_content_loss(output_feats[2], Variable(content_feat.data, requires_grad=False))
        loss_s = 0
        for i in range(4):
            loss_s += self.calc_style_loss(output_feats[i], style_feats[i].data)
        loss_t = self.calc_temporal_loss(g_t1, g_t2, flowout, mask)
        return loss_c, loss_s, loss_t

