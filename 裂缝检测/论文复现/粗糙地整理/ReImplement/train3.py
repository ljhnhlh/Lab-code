# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import math
import torch.nn as nn

class Crack(nn.Module):
    def __init__(self, Crack_cfg):
        super(Crack, self).__init__()
        self.features = self._make_layers(Crack_cfg)
        # linear layer
#         self.classifier = nn.Linear(512, 10)
#         self.linear1 = nn.Linear(32*6*6,64)
#         self.linear2 = nn.Linear(64,64)
#         self.linear3 = nn.Linear(64,25)
        self.classifier = self.make_classifier()
    
    def make_classifier(self):
        classifier = []
        classifier += [nn.Linear(32*6*6,64),nn.ReLU(inplace=True),nn.Dropout(p=0.5)]
        classifier += [nn.Linear(64,64),nn.ReLU(inplace=True),nn.Dropout(p=0.5)]
        classifier += [nn.Linear(64,2)]
        return nn.Sequential(*classifier)
    
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
#         out = self.linear1(out)
#         out = self.linear2(out)
#         out = self.linear3(out)
        return out

    def _make_layers(self, cfg):
        """
        cfg: a list define layers this layer contains
            'M': MaxPool, number: Conv2d(out_channels=number) -> BN -> ReLU
        """
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
            
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


Crack_cfg = {
    'Crack11':[16,16,'M',32,32,'M']
}

model_t = Crack(Crack_cfg['Crack11']);




plt.ion()   # interactive mode
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train' : transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]),
    "val" : transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
}

data_dir = 'I:\\1裂缝检测\\CrackForest-dataset\\trian\\train2\\'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])   
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=256,
                                             shuffle=True)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

# imshow(out, title=[class_names[x] for x in classes])
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # 这里可能是使用了模型库里面的模型？！
    best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0.0
    best_pr = 0.0
    best_re = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            L0 = 0
            P0 = 0
            P_neq = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                #L0 = TP + FN,即原本的裂缝数据，包括真裂缝和假非裂缝
                L0 += torch.sum(labels == 0)
                #P0 = TP+FP ,即预测为裂缝的数据，包括真裂缝和假裂缝
                P0 += torch.sum(preds == 0)
                #P1 = FN+TN,即预测为非裂缝的数据，包括真非裂缝和假非裂缝
                P_neq += torch.sum(preds != labels)
            
#             print(L0.numpy().size,P0.numpy().size,P1.numpy().size)
            t = L0+P0-P_neq
            print(t)
            print(2*P0)
            print(2*L0)
            #经计算，TP = (L0+P0-P1)/2
            #则PR = (L0+PO-P1)/(2*P0)
            PRECISE = t.float()/(2*P0)
            #RE = (L0+P0-P1)/(2*L0)，即原本的裂缝数据，有多少被检测出来了
            RECALL = t.float()/(2*L0)

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} PRE: {:.4f} REC: {:.4f}'.format(
                phase, epoch_loss, PRECISE,RECALL))
            # deep copy the model
            if phase == 'val' and (PRECISE >= best_pr or RECALL >= best_re):
                best_re = RECALL
                best_pr = PRECISE
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
  
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_re))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
best_pr = 0.0
best_re = 0.0
def train_model_25(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # 这里可能是使用了模型库里面的模型？！
    best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0.0
   

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            L0 = 0
            P0 = 0
            P_neq = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
#                     print(outputs.data.numpy().shape)
                    temp = np.array(outputs.data.numpy())
#                     print(np.sum(t[0]),np.sum(t[20]))
                    preds = []
                    for x in temp:
                        preds.append(np.sum(x))
                    preds = np.array(preds)
                    preds = (preds - np.min(preds))/(np.max(preds)-np.min(preds))
                    preds[preds>= 0.5] = 1
                    preds[preds < 0.5] = 0
#                     _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                labels = np.array(labels)
                #L0 = TP + FN,即原本的裂缝数据，包括真裂缝和假非裂缝
                L0 += np.sum(labels == 0)
                
                #P0 = TP+FP ,即预测为裂缝的数据，包括真裂缝和假裂缝
                P0 += np.sum(preds == 0)
                #P1 = FN+TN,即预测为非裂缝的数据，包括真非裂缝和假非裂缝
                P_neq += np.sum(preds != labels)
            
#             print(L0.numpy().size,P0.numpy().size,P1.numpy().size)
            t = L0+P0-P_neq
            print(t)
            print(2*P0)
            print(2*L0)
            #经计算，TP = (L0+P0-P1)/2
            #则PR = (L0+PO-P1)/(2*P0)
            PRECISE = t.float()/(2*P0)
            #RE = (L0+P0-P1)/(2*L0)，即原本的裂缝数据，有多少被检测出来了
            RECALL = t.float()/(2*L0)

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} PRE: {:.4f} REC: {:.4f}'.format(
                phase, epoch_loss, PRECISE,RECALL))
            # deep copy the model
            if phase == 'val' and (PRECISE >= best_pr or RECALL >= best_re):
                best_re = RECALL
                best_pr = PRECISE
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
  
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_re))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model


Crack_cfg = {
    'Crack11':[16,16,'M',32,32,'M']
}

model_ft = Crack(Crack_cfg['Crack11']);

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 7 epochs 退火
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
model_ft = model_ft.to('cpu')
torch.save(model_ft.cpu().state_dict(),"./crackre{:.4f}pr{:.4f}.pt".format(best_re,best_pr))
# torch.save(model.cpu().state_dict(),"./crackre{:.4f}pr{:.4f}.pt".format(best_re,best_pr))