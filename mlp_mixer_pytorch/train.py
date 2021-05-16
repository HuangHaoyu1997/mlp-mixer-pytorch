import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from mlp_mixer_pytorch import *
from einops.layers.torch import Rearrange

import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchsummary import summary
from thop import profile, clever_format
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize(0, 1) 
    ]
)
trainset = torchvision.datasets.CIFAR10(root='./mlp_mixer_pytorch/data/',train=True,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./mlp_mixer_pytorch/data/',train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=64,shuffle=False)

train_num = len(trainset)
test_num = len(testset)

model1 = MLPmixer(
    image_size = 32,
    patch_size = 4,
    dim = 128,
    depth = 4,
    num_classes = 10).to(device)

model2 = MLPMixer(
    image_size = 32,
    patch_size = 4,
    dim = 128,
    depth = 16,
    num_classes = 10,
    dropout=0.5).to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model2.parameters(),lr=1e-3)
optimizer = optim.SGD(model2.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
scheduler = lr_scheduler.StepLR(optimizer, step_size=15,gamma=0.1)

input = torch.randn(1, 3, 32, 32).to(device)
macs, params = profile(model2, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
print(macs,params)


for epoch in range(500):
    train_loss, test_loss = 0., 0.
    train_iter, test_iter = 0 , 0
    train_correct, test_correct = 0, 0
    # train
    scheduler.step()
    for inputs, labels in tqdm(trainloader):
        model2.train()
        # print(inputs.shape, labels.shape)
        optimizer.zero_grad()
        outputs = model2(inputs.to(device))
        pred = F.softmax(outputs.cpu()).argmax(dim=1)
        train_correct += torch.eq(pred,labels).float().sum().item()

        loss = criterion(outputs.cpu(),labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_iter += 1
    # test
    for inputs, labels in testloader:
        model2.eval()
        outputs = model2(inputs.to(device))
        pred = F.softmax(outputs.cpu()).argmax(dim=1)
        test_correct += torch.eq(pred,labels).float().sum().item()
        loss = criterion(outputs.cpu(),labels)
        test_loss += loss.item()
        test_iter += 1

    print('epoch:',epoch,'loss:',np.round(train_loss/train_iter,3),np.round(test_loss/test_iter,3),
            'acc:',train_correct/train_num,test_correct/test_num)