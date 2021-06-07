import os
import pdb
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

dir_data = 'dataset/hymenoptera_data'
model_name = 'ResNet'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dim_input = 224     # 图片尺寸
dim_output = 2      # 图片类别数
batch_size = 32
num_epochs = 2
train_mode = 'feature extraction'   # or 'fine tuning'


def init_model(model_name, train_mode, dim_output):
    model = None
    if model_name == 'ResNet':
        model = models.resnet18(pretrained=True if train_mode == 'feature extraction' else False)
        # for param in model.parameters():
        #     print(param.requires_grad)  # 均为True
        if train_mode == 'feature extraction':
            for param in model.parameters():
                param.requires_grad = False
        # 替换最后一层(fc)，进行我们自己的任务。前面的层数用作特征提取器
        dim_fc_input = model.fc.in_features
        model.fc = nn.Linear(dim_fc_input, dim_output)
        # print(model.fc.weight.requires_grad)    # True
    else:
        print('model not implemented')
    return model


def main():
    my_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(dim_input),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    # 已经证明并不是这种变换方式下各通道的mean和std
        ]),
        'val': transforms.Compose([
            transforms.Resize(dim_input),
            transforms.CenterCrop(dim_input),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    train_data = datasets.ImageFolder(
        os.path.join(dir_data, 'train'), my_transforms['train']
    )
    val_data = datasets.ImageFolder(
        os.path.join(dir_data, 'val'), my_transforms['val']
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = init_model(model_name=model_name, train_mode=train_mode, dim_output=dim_output)

    # for idx, (input, target) in enumerate(train_loader):
    #     # print(idx, input.shape, target.shape)   # 0 torch.Size([32, 3, 224, 224]) torch.Size([32])
    #     # 如果想要展示图片，记得不要对图片进行Normalize
    #     img = transforms.ToPILImage()(input[0])     # transforms的调用需要加括号
    #     plt.imshow(img)
    #     plt.pause(1)



if __name__ == '__main__':
    main()

