import os
import pdb
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from mod import ConvNet

'''testbench'''
'--- 展示数据'
# mnist_data = datasets.MNIST('dataset/mnist', download=True)
# mnist_data[0][0].show() # PIL图片可以直接show
# print(mnist_data[0][1])
'--- 将所有数据normalize所需的参数'
# mnist_data = datasets.MNIST('dataset/mnist', train=True, download=True, transform=transforms.Compose([
#     transforms.ToTensor(),
# ]))
# all_data = [d[0].data.numpy() for d in mnist_data]
# mean, std = np.mean(all_data), np.std(all_data)
# print(mean, std)    # 0.13066062 0.30810776



dir_model = 'model/ConvNet.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
lr = 0.01
momentum = 0.5
num_epochs = 1
loss_fun = F.nll_loss



def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for idx, (input, target) in enumerate(train_loader):
        # print(idx, (data.shape, target.shape))  # 0 (torch.Size([32, 1, 28, 28]), torch.Size([32]))
        input, target = input.to(device), target.to(device)
        output = model(input)   # (batch_size, 10)
        loss = loss_fun(output, target)
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 500 == 0:
            print(f'[TRAIN] Epoch: {epoch}, Iteration: {idx}, Loss: {loss}')


def test(model, device, test_loader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (input, target) in enumerate(test_loader):
            # print(idx, (data.shape, target.shape))  # 0 (torch.Size([32, 1, 28, 28]), torch.Size([32]))
            input, target = input.to(device), target.to(device)
            output = model(input)   # (batch_size, 10)
            total_loss += loss_fun(output, target, reduction='sum')
            pred = output.argmax(dim=1)     # (batch_size, 1)
            correct += pred.eq(target).sum().item()
    total_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset) * 100
    print(f'[TEST] Loss: {total_loss}, Accuracy: {acc}%')


def main():
    train_data = datasets.MNIST('dataset/mnist', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            0.13066062, 0.30810776
        )   # 预处理，对所有数据进行normalize
    ]))
    test_data = datasets.MNIST('dataset/mnist', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            0.13066062, 0.30810776
        )
    ]))
    # print(len(train_data), len(test_data))  # 60000, 10000
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size, shuffle=True, pin_memory=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size, shuffle=False, pin_memory=True
    )
    model = ConvNet().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr, momentum=momentum
    )
    if os.path.exists(dir_model):
        model.load_state_dict(torch.load(dir_model))
    else:
        for epoch in range(num_epochs):
            train(model, device, train_loader, optimizer, epoch)
        os.makedirs('model', exist_ok=True)
        torch.save(model.state_dict(), dir_model)
    test(model, device, test_loader)

if __name__ == '__main__':
    main()