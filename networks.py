import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cbam import CBAM
from torchvision.models import vgg19

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
    def forward(self, x):
        x = self.conv(x)
        return F.relu(x, inplace=True)
class inception_module(nn.Module):
    # referred from https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
    def __init__(self, in_channels, pool_features):
        super(inception_module, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)
        self.conv2d = nn.Conv2d(225, 1, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        outputs = torch.cat(outputs, 1)
        outputs = self.conv2d(outputs)
        return outputs

class WGAN_VGG_FeatureExtractor(nn.Module):
    def __init__(self):
        super(WGAN_VGG_FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])
    def forward(self, x):
        out = self.feature_extractor(x)
        return out

class WGAN_VGG_generator(nn.Module):
    def __init__(self, out_ch=64):
        super(WGAN_VGG_generator, self).__init__()
        self.conv_first = nn.Conv2d(1, out_ch, kernel_size=5, stride=5, padding=0)  # kernel_size=3,bias=False
        self.inception_module = inception_module(in_channels=1,pool_features=1)
        self.CBAM = CBAM(gate_channels=10)
        self.LeakyReLU = nn.LeakyReLU(0.1, inplace=True)  # nn.ReLU
        self.conv_1 = nn.Conv2d(1, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_last = nn.Conv2d(out_ch, 1, kernel_size=1, stride=1, padding=0)
        # self.feature_extractor = WGAN_VGG_FeatureExtractor()

    def forward(self, x, depth=7, ratio=4, width=64, alpha=0.1):
        out = self.LeakyReLU(self.conv_first(x))
        for i in range(depth):
            block_in = x
            out = self.inception_module(x)
            out = self.CBAM(out)
            out += block_in
        out = self.LeakyReLU(self.conv_1(out))
        out = self.conv_last(out)
        return out

class WGAN_VGG_discriminator(nn.Module):
    def __init__(self, input_size):
        super(WGAN_VGG_discriminator, self).__init__()
        def conv_output_size(input_size, kernel_size_list, stride_list):
            n = (input_size - kernel_size_list[0]) // stride_list[0] + 1
            for k, s in zip(kernel_size_list[1:], stride_list[1:]):
                n = (n - k) // s + 1
            return n
        def add_block(layers, ch_in, ch_out, stride):
            layers.append(nn.Conv2d(ch_in, ch_out, 3, stride, 0))
            layers.append(nn.Tanh())
            return layers
        layers = []
        ch_stride_set = [(1, 64, 1), (64, 64, 2), (64, 128, 1), (128, 128, 2), (128, 256, 1), (256, 256, 2)]
        for ch_in, ch_out, stride in ch_stride_set:
            add_block(layers, ch_in, ch_out, stride)
        self.output_size = conv_output_size(input_size, [3]*6, [1,2]*3)
        self.net = nn.Sequential(*layers)
        self.fc1 = nn.Linear(256*self.output_size*self.output_size, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        out = self.net(x)
        out = out.view(-1, 256*self.output_size*self.output_size)
        out = self.lrelu(self.fc1(out))
        out = self.fc2(out)
        return out

class WGAN_VGG(nn.Module):
    # referred from https://github.com/kuc2477/pytorch-wgan-gp
    def __init__(self, input_size=64, gate_channels=10):
        super(WGAN_VGG, self).__init__()
        self.generator = WGAN_VGG_generator()
        self.discriminator = WGAN_VGG_discriminator(input_size)
        self.cbam = CBAM(gate_channels)
        self.p_criterion = nn.MSELoss()
    def d_loss(self, x, y, gp=True, return_gp=False):
        fake = self.generator(x)
        d_real = self.discriminator(y)
        d_fake = self.discriminator(fake)
        d_loss = -torch.mean(d_real) + torch.mean(d_fake)
        if gp:
            gp_loss = self.gp(y, fake)
            loss = d_loss + gp_loss
        else:
            gp_loss = None
            loss = d_loss
        return (loss, gp_loss) if return_gp else loss

    def g_loss(self, x, y):
        g_critic = nn.L1Loss()
        fake = self.generator(x)
        loss = g_critic(fake, y)
        return loss

    def gp(self, y, fake, lambda_=10):
        assert y.size() == fake.size()
        a = torch.cuda.FloatTensor(np.random.random((y.size(0), 1, 1, 1)))
        interp = (a*y + ((1-a)*fake)).requires_grad_(True)
        d_interp = self.discriminator(interp)
        fake_ = torch.cuda.FloatTensor(y.shape[0], 1).fill_(1.0).requires_grad_(False)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interp, grad_outputs=fake_,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean() * lambda_
        return gradient_penalty
