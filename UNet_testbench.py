import os
import pdb
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter

from mod import UNet_Dataset, UNet




def train(model, dataloader, writer, device, optimizer, loss_fun, epoch):
    model.train()
    avg_loss = 0
    for idx, (input, label) in enumerate(dataloader):
        if not idx:
           batches_loss = 0
        optimizer.zero_grad()
        input, label = input.to(device), label.to(device)
        output = model(input)
        loss = loss_fun(output, label)
        batches_loss += loss
        avg_loss += loss
        loss.backward()
        optimizer.step()

        if idx and idx % 5 == 0:
            # writer.add_scalar(
            #     'training loss', batches_loss/5, epoch * len(dataloader) + idx
            # )
            print('[TRAIN] Epoch: {}, Iteration: {}, Loss: {}'.format(epoch, idx, batches_loss/5))
            batches_loss = 0
    return avg_loss / len(dataloader)
def main():
    dir_TB = 'TB/UNet'
    writer = None
    # writer = SummaryWriter(dir_TB)
    dir_input = 'dataset/UNet/input'
    dir_GT = 'dataset/UNet/GT'
    dataset = UNet_Dataset(dir_input, dir_GT)
    batch_size = 32
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    num_epochs = 10
    dir_model = 'model/UNet'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fun = F.mse_loss
    loss_ls = []    # for visualization
    # if os.path.exists(dir_model):
    #     model.load_state_dict(torch.load(dir_model + 'UNet_{}.pt'.format(num_epoch-1)))
    # else:
    if not os.path.exists(dir_model):
        os.makedirs(dir_model)
    for epoch in range(num_epochs):
        loss = train(model, dataloader, writer, device, optimizer, loss_fun, epoch)
        if epoch and epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(dir_model, 'UNet_{}.pt'.format(epoch)))
            loss_ls.append(loss)


    plt.title('LOSS / EPOCH')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(len(loss_ls), loss_ls, label="loss_ls")
    plt.ylim((0, 1.))
    plt.legend()
    plt.show()
    plt.savefig('UNet1')

if __name__ == '__main__':
    main()