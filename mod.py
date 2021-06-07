import os

import cv2 as cv
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms




'--- ConvNet'
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # (batch_size, 1, 28, 28) for MNIST
        x = F.relu(self.conv1(x))
        # (batch_size, 20, 24, 24)
        x = F.max_pool2d(x, 2, 2)
        # (batch_size, 20, 12, 12)
        x = F.relu(self.conv2(x))
        # (batch_size, 50, 8, 8)
        x = F.max_pool2d(x, 2, 2)
        # (batch_size, 50, 4, 4)
        x = x.view(-1, 50*4*4)
        # (batch_size, 800)
        x = F.relu(self.fc1(x))
        # (batch_size, 500)
        x = self.fc2(x)
        # (batch_size, 10)
        x = F.log_softmax(x, dim=1)
        # (batch_size, 10)
        return x


'--- UNet'
class UNet_Dataset(Dataset):
    def __init__(self, dir_input, dir_GT):
        super(UNet_Dataset, self).__init__()
        self.dir_input = dir_input
        self.dir_GT = dir_GT
        self.files = os.listdir(self.dir_input)
        self.trans = transforms.Compose((
            transforms.ToTensor(),
        ))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        file_name = self.files[item]
        input = cv.imread(os.path.join(self.dir_input, file_name))
        GT = cv.imread(os.path.join(self.dir_GT, file_name))
        return self.trans(input), self.trans(GT)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.layer(x)

class Down_DC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down_DC, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.layer(x)

class Up_DC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up_DC, self).__init__()
        self.dim_output = out_channels
        self.layer1 = F.interpolate
        self.layer2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.layer3 = DoubleConv(in_channels, out_channels)
    def forward(self, x, x_shortcut):
        x = self.layer1(x, scale_factor=2)
        x = self.layer2(x)
        x = torch.cat((x_shortcut, x), dim=1)
        x = self.layer3(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.inc = DoubleConv(3, 64)
        self.down1 = Down_DC(64, 128)
        self.down2 = Down_DC(128, 256)
        self.down3 = Down_DC(256, 512)
        self.down4 = Down_DC(512, 1024)
        self.up4 = Up_DC(1024, 512)
        self.up3 = Up_DC(512, 256)
        self.up2 = Up_DC(256, 128)
        self.up1 = Up_DC(128, 64)
        self.out = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.out(x)
        return x

