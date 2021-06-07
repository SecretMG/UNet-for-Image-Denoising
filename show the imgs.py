import os
import cv2 as cv
import numpy as np

import torch
from torch.utils.data import DataLoader
from UNet_testbench import UNet_Dataset, UNet
from torchvision import transforms



def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img



dir_model = 'model/UNet'
dir_model0 = os.path.join(dir_model, 'UNet_{}.pt'.format(0))
dir_model50 = os.path.join(dir_model, 'UNet_{}.pt'.format(50))
dir_model100 = os.path.join(dir_model, 'UNet_{}.pt'.format(100))
dir_model150 = os.path.join(dir_model, 'UNet_{}.pt'.format(150))
dir_model200 = os.path.join(dir_model, 'UNet_{}.pt'.format(200))
dir_model250 = os.path.join(dir_model, 'UNet_{}.pt'.format(250))
dir_models = [dir_model0, dir_model50, dir_model100, dir_model150, dir_model200, dir_model250]


models = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for dir_model in dir_models:
    model = UNet().to(device)
    model.load_state_dict(torch.load(dir_model, map_location=device))
    models.append(model)


dir_input = 'dataset/UNet/input'
dir_GT = 'dataset/UNet/GT'
dataset = UNet_Dataset(dir_input, dir_GT)
batch_size = 1
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=False
)
model.eval()
for input, label in dataloader:
    for idx, model in enumerate(models):
        output = model(input)
        output = tensor_to_np(output)
        cv.imshow('output{}'.format(idx), output)
        cv.resizeWindow('output{}'.format(idx), 500, 300)
    input = tensor_to_np(input)
    cv.imshow('input', input)
    label = tensor_to_np(label)
    cv.imshow('label', label)
    cv.waitKey()