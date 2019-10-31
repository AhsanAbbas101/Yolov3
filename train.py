# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 01:13:23 2019

@author: Ahsan
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable

from dataset import *
from darknet import DarkNet

#torch.set_default_tensor_type('torch.cuda.FloatTensor')

dataset_trainset = IIITDataset(csv_file="annotations.csv", root_dir="data/images/")
#mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.ToTensor())
#mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform = transforms.ToTensor())

train_loader = DataLoader(dataset=dataset_trainset, batch_size=32, shuffle=False)
#test_loader = DataLoader(dataset=mnist_testset, batch_size=1000, shuffle=True)


model = DarkNet()
model = model.cuda()


# optimizer = optim.Adam(model.parameters(), lr=0.0001)
optimizer = optim.Adam(model.parameters())

#print(list(enumerate(train_loader, 1)))

def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(train_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        
        input = input.cuda()
        model.target = target
        
        #DO forward_call
        output, loss = model.forward(input)
        
        
        optimizer.zero_grad()
        #loss = criterion(out, input)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    print("===> Epoch {} Complete: Avg. Loss: {:.6f}".format(epoch, epoch_loss / len(train_loader)))
    
for epoch in range(1, 25):
    train(epoch)
    
torch.save(model.state_dict(), "weights.pt")
