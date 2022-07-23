import torch
import numpy as np
import torch
from network.ConvLSTM import ConvLSTM
from torch.nn.parameter import Parameter

class InstanceBank(torch.nn.Module):
    def __init__(self, out_channels, banknumber):
        super(InstanceBank, self).__init__()
        self.norm = torch.nn.InstanceNorm2d(out_channels, affine=False, momentum=1.0)
        for i in range(120):
            setattr(self, 'instance{}'.format(i), torch.nn.InstanceNorm2d(out_channels, affine=True, momentum=1.0))
            
    def forward(self, x, bank, alpha=1.0):
        if bank == -1:
            return x
        return getattr(self, 'instance{}'.format(bank))(x) * alpha + (1-alpha) * self.norm(x)




class SelectiveLoadModule(torch.nn.Module):
    """Only load layers in trained models with the same name."""
    def __init__(self):
        super(SelectiveLoadModule, self).__init__()

    def forward(self, x):
        return x

    def load_state_dict(self, state_dict):
        """Override the function to ignore redundant weights."""
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param)


class ConvLayer(torch.nn.Module):
    """Reflection padded convolution layer."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ConvTanh(ConvLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvTanh, self).__init__(in_channels, out_channels, kernel_size, stride)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        out = super(ConvTanh, self).forward(x)
        out = (self.tanh(out) + 1.) / 2.
        return out


class ConvInstRelu(ConvLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, banks=10):
        super(ConvInstRelu, self).__init__(in_channels, out_channels, kernel_size, stride, banks)
        self.instance = InstanceBank(out_channels, banks)
        self.relu = torch.nn.ReLU()

    def forward(self, x, bank):
        out = super(ConvInstRelu, self).forward(x)
        out = self.instance(out, bank)
        out = self.relu(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    """Upsamples the input and then does a convolution.
    This method gives better results compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class UpsampleConvInstRelu(UpsampleConvLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None, banks=10):
        super(UpsampleConvInstRelu, self).__init__(in_channels, out_channels, kernel_size, stride, upsample)
        self.instance = InstanceBank(out_channels, banks)
        self.relu = torch.nn.ReLU()

    def forward(self, x, bank):
        out = super(UpsampleConvInstRelu, self).forward(x)
        out = self.instance(out, bank)
        out = self.relu(out)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, banks=10):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride)
        self.in1 = InstanceBank(out_channels, banks)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, stride)
        self.in2 = InstanceBank(out_channels, banks)
        self.relu = torch.nn.ReLU()

    def forward(self, x, bank):
        residual = x
        out = self.relu(self.in1(self.conv1(x), bank))
        out = self.in2(self.conv2(out), bank)
        out = out + residual
        return out


class ReCoNet(torch.nn.Module):
    def __init__(self):
        super(ReCoNet, self).__init__()

        self.style_conv1 = ConvInstRelu(3, 32, kernel_size=9, stride=1)
        self.style_conv2 = ConvInstRelu(32, 64, kernel_size=3, stride=2)
        self.style_conv3 = ConvInstRelu(64, 128, kernel_size=3, stride=2)

        self.style_res1 = ResidualBlock(128, 128)
        self.style_res2 = ResidualBlock(128, 128)
        self.style_res3 = ResidualBlock(128, 128)
        self.style_res4 = ResidualBlock(128, 128)
        self.style_res5 = ResidualBlock(128, 128)

        self.style_deconv1 = UpsampleConvInstRelu(128, 64, kernel_size=3, stride=1, upsample=2)
        self.style_deconv2 = UpsampleConvInstRelu(64, 32, kernel_size=3, stride=1, upsample=2)
        self.style_deconv3 = ConvTanh(32, 3, kernel_size=9, stride=1)
        self.convlstm1 = ConvLSTM(input_size=128, hidden_size=128, kernel_size=3)
        self.convlstm2 = ConvLSTM(input_size=128, hidden_size=128, kernel_size=3)

        self.catconv1 = ConvInstRelu(256, 128, kernel_size=3, stride=1)
        self.catconv2 = ConvInstRelu(256, 128, kernel_size=3, stride=1)


    def encoder(self, x, bank):
        x = self.style_conv1(x, bank)
        x = self.style_conv2(x, bank)
        x = self.style_conv3(x, bank)
        return x

    def decoder(self, x, bank):
        x = self.style_deconv1(x, bank)
        x = self.style_deconv2(x, bank)
        x = self.style_deconv3(x)
        return x

    def transformer(self, x, bank):
        x = self.style_res1(x, bank)
        x = self.style_res2(x, bank)
        x = self.style_res3(x, bank)
        x = self.style_res4(x, bank)
        x = self.style_res5(x, bank)
        return x


    def forward2lstm(self, x, prev_state1, prev_state2, bank=0):
        x = self.encoder(x, bank)
        prev_state1 = self.convlstm1(x, prev_state1)
        z = torch.cat((prev_state1[0], x), 1)
        z = self.catconv1(z, bank)

        z = self.transformer(z, bank)
        prev_state2 = self.convlstm2(z, prev_state2)
        z = torch.cat((prev_state2[0], z), 1)
        z = self.catconv2(z, bank)
        return self.decoder(z, bank), prev_state1, prev_state2

    def forward(self, x, prev_state1, prev_state2, bank=0, pos='up'):
        if pos == 'up':
            return self.forward1(x, prev_state1, prev_state2, bank)
        if pos == 'down':
            return self.forward2(x, prev_state1, prev_state2, bank)
        if pos == 'middle':
            return self.forward3(x, prev_state1, prev_state2, bank)

        return self.forward2lstm(x, prev_state1, prev_state2, bank)


    def forward1(self, x, prev_state1, prev_state2, bank=0):
        x = self.encoder(x, bank)
        z = x
        z = self.transformer(z, bank)

        prev_state2 = self.convlstm2(z, prev_state2)
        z = torch.cat((prev_state2[0], z), 1)
        z = self.catconv2(z, bank)

        return self.decoder(z, bank), prev_state1, prev_state2


    def forward2(self, x, prev_state1, prev_state2, bank=0):
        x = self.encoder(x, bank)

        prev_state1 = self.convlstm1(x, prev_state1)
        z = torch.cat((prev_state1[0], x), 1)
        z = self.catconv1(z, bank)

        z = self.transformer(z, bank)
        return self.decoder(z, bank), prev_state1, prev_state2


    def forward3(self, x, prev_state1, prev_state2, bank=0):
        x = self.encoder(x, bank)

        z,  prev_state1 = self.transformer_middle(x, prev_state1, bank)

        return self.decoder(z, bank), prev_state1, prev_state2


