# from __future__ import division, print_function

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, mode_upsampling=1):
        super(UpBlock, self).__init__()
        self.mode_upsampling = mode_upsampling
        if mode_upsampling == 0:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        elif mode_upsampling == 1:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif mode_upsampling == 2:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        elif mode_upsampling == 3:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.mode_upsampling != 0:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)# 16x256x256
        x1 = self.down1(x0) # 32x128x128
        x2 = self.down2(x1) # 64x64x64
        x3 = self.down3(x2) # 128x32x32
        x4 = self.down4(x3) # 256x16x16
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0,
                           mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0] # 16x256x256
        x1 = feature[1] # 32x128x128
        x2 = feature[2] # 64x64x64
        x3 = feature[3] # 128x32x32
        x4 = feature[4] # 256x16x16

        x1_3 = self.up1(x4, x3)    # 128x32x32
        x1_2 = self.up2(x1_3, x2) # 64x64x64
        x1_1 = self.up3(x1_2, x1) # 32x128x128
        x1_0 = self.up4(x1_1, x0) # 16x256x256
        output = self.out_conv(x1_0)   # 1x256x256
        return x1_3, x1_2, x1_1, x1_0, output


class SideConv(nn.Module):
    def __init__(self, n_classes=4):
        super(SideConv, self).__init__()

        self.side5 = nn.Conv2d(128, n_classes, 1, padding=0)
        self.side4 = nn.Conv2d(64, n_classes, 1, padding=0)
        self.side3 = nn.Conv2d(32, n_classes, 1, padding=0)
        self.side2 = nn.Conv2d(16, n_classes, 1, padding=0)
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, stage_feat):
        x5, x5_up, x6_up, x7_up = stage_feat[0], stage_feat[1], stage_feat[2], stage_feat[3]    # # 128x32x32，64x64x64，32x128x128，16x256x256
        out5 = self.side5(x5)
        out5 = self.upsamplex2(out5)
        out5 = self.upsamplex2(out5)
        out5 = self.upsamplex2(out5)

        out4 = self.side4(x5_up)
        out4 = self.upsamplex2(out4)
        out4 = self.upsamplex2(out4)

        out3 = self.side3(x6_up)
        out3 = self.upsamplex2(out3)

        out2 = self.side2(x7_up)

        return [out5, out4, out3, out2]

class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params1 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 1,
                   'acti_func': 'relu'}

        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)

    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        return output1


class DDNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(DDNet, self).__init__()

        params1 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 1,
                   'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 0,
                   'acti_func': 'relu'}
        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)
        self.sideconv1 = SideConv()

    def forward(self, x):
        feature = self.encoder(x)
        x1_3_1, x1_2_1, x1_1_1, x1_0_1, output1 = self.decoder1(feature)
        x2_3_2, x2_2_2, x2_1_2, x2_0_2, output2 = self.decoder2(feature)
        stage_feat1 = [x1_3_1, x1_2_1, x1_1_1, x1_0_1]
        stage_feat2 = [x2_3_2, x2_2_2, x2_1_2, x2_0_2]
        deep_out1 = self.sideconv1(stage_feat1)
        deep_out2 = self.sideconv1(stage_feat2)
        return  [output1, output2],deep_out2


if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    # from ptflops import get_model_complexity_info

    model = DDNet(in_chns=3, class_num=1)

    total = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter: % .2fM' % (total / 1e6))

    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(model, (1, 256, 256), as_strings=True,
    #                                              print_per_layer_stat=True, verbose=True)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # import ipdb;
    #
    # ipdb.set_trace()
