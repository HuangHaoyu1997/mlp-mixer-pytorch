import torch
import torch.nn as nn
from mlp_mixer_pytorch import *
from einops.layers.torch import Rearrange

import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
trainset = torchvision.datasets.CIFAR10(root='./mlp_mixer_pytorch/data/',train=True,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./mlp_mixer_pytorch/data/',train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=False)

model1 = MLPmixer(
    image_size = 32,
    patch_size = 4,
    dim = 128,
    depth = 4,
    num_classes = 10
).to(device)
model2 = MLPMixer(
    image_size = 32,
    patch_size = 4,
    dim = 128,
    depth = 4,
    num_classes = 10
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model2.parameters(),lr=1e-3)


'''
img = torch.randn(1, 3, 256, 256)
pred1 = model1(img) # (1, 1000)
pred2 = model2(img) # (1, 1000)

print(pred1[0,0:8])
print(pred2[0,0:8])
'''

for epoch in range(10):
    running_loss = 0.
    iteration = 0
    for inputs, labels in trainloader:
        # print(inputs.shape, labels.shape)
        optimizer.zero_grad()
        outputs = model2(inputs.to(device))

        loss = criterion(outputs.cpu(),labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        iteration += 1

    print(running_loss/iteration)