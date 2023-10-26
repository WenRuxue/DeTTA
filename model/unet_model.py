""" Full assembly of the parts to form the complete network """
import torch

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1 = Up(512, 256, bilinear = False)
        self.up2 = Up(256, 128, bilinear = False)
        self.up3 = Up(128, 64, bilinear = False)
        self.up4 = Up(64, 32, bilinear = False)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4, flag = False)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        pred_y = torch.sigmoid(logits)
        return pred_y

class DeYNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(DeYNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1_mask = Up(512, 256, bilinear = False)
        self.up2_mask = Up(256, 128, bilinear = False)
        self.up3_mask = Up(128, 64, bilinear = False)
        self.up4_mask = Up(64, 32, bilinear = False)
        self.up1_de = Up(512, 256, bilinear=False)
        self.up2_de = Up(256, 128, bilinear=False)
        self.up3_de = Up(128, 64, bilinear=False)
        self.up4_de = Up(64, 32, bilinear = False)
        self.outc_mask = OutConv(32, n_classes)
        self.outc_de = OutConv(32, n_classes)

    def forward(self, x, adaptive=False):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1_mask(x5, x4, flag = False)
        x = self.up2_mask(x, x3)
        x = self.up3_mask(x, x2)
        x = self.up4_mask(x, x1)
        logits = self.outc_mask(x)
        pred_mask = torch.sigmoid(logits)
        z = self.up1_de(x5, x4, flag=False)
        z = self.up2_de(z, x3)
        z = self.up3_de(z, x2)
        z = self.up4_de(z, x1)
        logits = self.outc_de(z)
        denoised_image = torch.sigmoid(logits)

        return pred_mask,denoised_image

class UNet_rotation(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_rotation, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1_mask = Up(512, 256, bilinear = False)
        self.up2_mask = Up(256, 128, bilinear = False)
        self.up3_mask = Up(128, 64, bilinear = False)
        self.up4_mask = Up(64, 32, bilinear = False)
        self.outc_mask = OutConv(32, n_classes)
        self.avgpool = nn.AvgPool2d(32)
        self.fc = nn.Linear(256, 4)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1_mask(x5, x4, flag = False)
        x = self.up2_mask(x, x3)
        x = self.up3_mask(x, x2)
        x = self.up4_mask(x, x1)
        logits = self.outc_mask(x)
        pred_mask = torch.sigmoid(logits)
        z = self.avgpool(x5)
        z = z.view(z.size(0), -1)
        z = self.fc(z)
        return pred_mask,z