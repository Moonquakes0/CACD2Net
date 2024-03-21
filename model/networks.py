import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from util.util import initialize_weights

def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    def __init__(
            self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.999))

        if relu:
            self.add_module("relu", nn.ReLU())


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
            解释  :
                bmm : 实现batch的叉乘
                Parameter：绑定在层里，所以是可以更新的
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class DARNetHead(nn.Module):
    def __init__(self, in_channels):
        super(DARNetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        sa_feat = self.conv5a(x)
        sa_feat = self.sa(sa_feat)
        sa_feat = self.conv51(sa_feat)

        sc_feat = self.conv5c(x)
        sc_feat = self.sc(sc_feat)
        sc_feat = self.conv52(sc_feat)

        # 两个注意力是相加的
        feat_sum = sa_feat + sc_feat

        output = self.dropout(feat_sum)
        return output

class Featrue_Fusion(nn.Module):
    def __init__(self, dim):
        super(Featrue_Fusion, self).__init__()
        self.expand = add_conv(dim, 64, 3, 1,leaky=False)

        compress_c = 8   #when adding rfb, we use half number of channels to save memory


        self.weight_level_1 = add_conv(dim, compress_c, 1, 1, leaky=False)
        self.weight_level_2 = add_conv(dim, compress_c, 1, 1, leaky=False)
        self.weight_level_3 = add_conv(dim, compress_c, 1, 1, leaky=False)
        self.weight_level_4 = add_conv(dim, compress_c, 1, 1, leaky=False)

        self.weight_levels = nn.Conv2d(compress_c*4, 4, kernel_size=1, stride=1, padding=0)


    def forward(self, x4, x3, x2, x1):


        level_1_weight_v = self.weight_level_1(x1)
        level_2_weight_v = self.weight_level_2(x2)
        level_3_weight_v = self.weight_level_3(x3)
        level_4_weight_v = self.weight_level_4(x4)
        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v, level_4_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        fusion = x1 * levels_weight[:, 0:1, :, :] + x2 * levels_weight[:, 1:2, :, :] + x3 * levels_weight[:, 2:3, :, :] + x4 * levels_weight[:, 3:, :, :]

        return fusion

class Guided_Map_Generator_res(nn.Module):
    def  __init__(self, nclass):
        super(Guided_Map_Generator_res, self).__init__()
        self.head = DARNetHead(64)
        self.fusion = Featrue_Fusion(64)
        self.expand = add_conv(64, 64, 3, 1, leaky=False)

        self.last_conv = nn.Sequential(
                                       nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(16, nclass, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, layer4, layer3, layer2, layer1):

        layer1 = F.upsample(layer1, layer4.size()[2:], mode='bilinear', align_corners=True)
        Fusion_feature = self.fusion(layer4, layer3, layer2, layer1)
        Fusion_feature = self.expand(Fusion_feature)
        x1_head = self.head(Fusion_feature)
        out = self.last_conv(x1_head)

        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class GAM(nn.Module):
    def __init__(self):
        super(GAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        B, C, H, W = x1.size()
        x2_cat = x2
        for i in range(C-1):
            x2_cat = torch.cat((x2,x2_cat),dim=1)
        x1_view = x1.view(B,C,-1)
        x2_view = x2_cat.view(B,C,H*W).permute(0,2,1)
        energy = torch.bmm(x1_view, x2_view)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        relation = self.softmax(energy_new)
        x1_new = x1.view(B, C, -1)
        out = torch.bmm(relation,x1_new)
        out = out.view(B, C, H, W)
        out = self.gamma * out +x1

        return out

class DEM(nn.Module):
    def __init__(self, in_dim_low, nclasses):
        super(DEM, self).__init__()
        self.seg_map = nn.Sigmoid()
        self.classes = nclasses
        self.low_channel = in_dim_low
        self.out = nn.Sequential(
            nn.Conv2d(self.low_channel, self.low_channel, 3, 1, 1),
            nn.Conv2d(self.low_channel, self.classes, 3, 1, 1))

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.bn1 = nn.BatchNorm2d(self.low_channel)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.low_channel)
        self.relu2 = nn.ReLU()

    def forward(self, low_feature, x):

        seg_prob = self.seg_map(x)
        foreground = low_feature * seg_prob
        background = low_feature * (1 - seg_prob)
        refine1 = self.alpha * foreground
        refine1 = self.bn1(refine1)
        refine1 = self.relu1(refine1)
        refine2 = self.beta * background
        refine2 = self.bn2(refine2)
        refine2 = self.relu2(refine2)
        fusion = refine1 - refine2
        output_map = self.out(fusion)

        return output_map

class refinment_decoder(nn.Module):
    def __init__(self, in_dim_low, nclasses):
        super(refinment_decoder, self).__init__()
        self.seg_map = nn.Sigmoid()
        self.cgm = GAM()
        self.dem = DEM(in_dim_low,nclasses)
        self.classes = nclasses
        self.low_channel = in_dim_low
        self.out = nn.Sequential(
            nn.Conv2d(self.low_channel, self.low_channel, 3, 1, 1),
            nn.Conv2d(self.low_channel, self.classes, 3, 1, 1))

    def forward(self, low_feature, x):
        low_feature = F.upsample(low_feature, x.size()[2:], mode='bilinear', align_corners=True)
        low_feature = self.cgm(low_feature,x)
        output = self.dem(low_feature, x)

        return output


class DFRM(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(DFRM, self).__init__()
        self.relu = nn.ReLU(True)

        self.conv1 = _ConvBnReLU(dim_in, dim_in, 3, 1, 1, 1)
        self.conv2 = _ConvBnReLU(dim_in, dim_in, 3, 1, 1, 1)

        self.branch1 = _ConvBnReLU(dim_in, dim_out//4, 3, 1, 1, 1)
        self.branch2 = _ConvBnReLU(dim_in, dim_out//4, 3, 1, 4, 4)
        self.branch3 = _ConvBnReLU(dim_in, dim_out//4, 3, 1, 8, 8)
        self.branch4 = _ConvBnReLU(dim_in, dim_out//4, 3, 1, 16, 16)
        self.res = _ConvBnReLU(dim_in, dim_out, 1, 1, 0, 1)


        self.cat = _ConvBnReLU(dim_out * 3, dim_out, 3, 1, 1, 1)


    def forward(self, x1, x2):
        x_add = x1 + x2
        x_diff = torch.abs(x1 - x2)
        y = self.conv1(x_diff) + self.conv2(x_add)
        y1 = self.branch1(y)
        y2 = self.branch2(y)
        y3 = self.branch3(y)
        y4 = self.branch4(y)

        y = self.cat(torch.cat([self.res(x1), self.res(x2), y1, y2, y3, y4], 1))
        return y



class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels > 3:
            newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels - 3, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)



class cacd2net(nn.Module):
    def __init__(self, in_channels=3):
        super(cacd2net, self).__init__()
        self.FCN = FCN(in_channels, pretrained=True)
        self.coarse_out = Guided_Map_Generator_res(1)
        self.layer4_process = DFRM(512, 64)
        self.layer3_process = DFRM(256, 64)
        self.layer2_process = DFRM(128, 64)
        self.layer1_process = DFRM(64, 64)

        self.out_1 = refinment_decoder(64, 1)
        self.out_2 = refinment_decoder(64, 1)

    def fcn_forward(self, x):

        fea0 = self.FCN.layer0(x)  # size:1/4
        fea0 = self.FCN.maxpool(fea0)  # size:1/4
        fea1 = self.FCN.layer1(fea0)  # size:1/4
        fea2 = self.FCN.layer2(fea1)  # size:1/8
        fea3 = self.FCN.layer3(fea2)  # size:1/16
        fea4 = self.FCN.layer4(fea3)


        return fea1, fea2, fea3, fea4


    def forward(self, x1, x2):
        x_size = x1.size()

        fea1_1, fea2_1, fea3_1, fea4_1 = self.fcn_forward(x1)
        fea1_2, fea2_2, fea3_2, fea4_2 = self.fcn_forward(x2)

        layer4 = self.layer4_process(fea4_1, fea4_2)
        layer3 = self.layer3_process(fea3_1,fea3_2)
        layer2 = self.layer2_process(fea2_1,fea2_2)
        layer1 = self.layer1_process(fea1_1,fea1_2)

        coarse_out = self.coarse_out(layer4, layer3, layer2, layer1)

        layer2_process = F.upsample(layer2, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer3_process = F.upsample(layer3, layer1.size()[2:], mode='bilinear', align_corners=True)

        layer_1_2_3 = layer1+layer2_process+layer3_process

        result_1 = self.out_1(layer_1_2_3, coarse_out)
        result_2 = self.out_2(layer_1_2_3, result_1)


        return F.upsample(coarse_out, x_size[2:], mode='bilinear'),F.upsample(result_1, x_size[2:], mode='bilinear'),F.upsample(result_2, x_size[2:], mode='bilinear')