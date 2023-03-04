import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

NUM_CLASSES = 21

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size()[0], 16 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 5)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, 3)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 16, 3)
        self.bn5 = nn.BatchNorm2d(16)
        self.conv1 = nn.Conv2d(3, 128, 5)
        
        # self.conv2 = nn.Conv2d(128, 64, 1)
        # self.conv3 = nn.Conv2d(64, 64, 3)
        # self.conv4 = nn.Conv2d(64, 128, 1, stride=1, padding=1)
        
        # self.conv5 = nn.Conv2d(128, 64, 1)
        # self.conv6 = nn.Conv2d(64, 64, 3)
        # self.conv7 = nn.Conv2d(64, 128, 1, stride=1, padding=1)
        
        # self.conv8 = nn.Conv2d(128, 64, 1)
        # self.conv9 = nn.Conv2d(64, 64, 3)
        # self.conv10 = nn.Conv2d(64, 128, 1, stride=1, padding=1)
        
        # self.conv11 = nn.Conv2d(128, 16, 3)
        
        # self.bn1 = nn.BatchNorm2d(128)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.bn4 = nn.BatchNorm2d(128)
        # self.bn5 = nn.BatchNorm2d(64)
        # self.bn6 = nn.BatchNorm2d(64)
        # self.bn7 = nn.BatchNorm2d(128)
        # self.bn8 = nn.BatchNorm2d(64)
        # self.bn9 = nn.BatchNorm2d(64)
        # self.bn10 = nn.BatchNorm2d(128)
        # self.bn11 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 25 * 25, 120)
        # self.fc1 = nn.Linear(16 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)
        return

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = self.pool(F.relu(x))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = x.view(x.size()[0], 16 * 25 * 25)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.relu(self.bn1(self.conv1(x)))
        
        # res = x
        # out = F.relu(self.bn2(self.conv2(x)))
        # out = F.relu(self.bn3(self.conv3(out)))
        # out = F.relu(self.bn4(self.conv4(out)))
        # out += res
        # out = self.pool(out)
        
        # res = out
        # out2 = F.relu(self.bn5(self.conv5(out)))
        # out2 = F.relu(self.bn6(self.conv6(out2)))
        # out2 = F.relu(self.bn7(self.conv7(out2)))
        # out2 += res
        # out2 = self.pool(out2)
        
        # res = out2
        # out3 = F.relu(self.bn8(self.conv8(out2)))
        # out3 = F.relu(self.bn9(self.conv9(out3)))
        # out3 = F.relu(self.bn10(self.conv10(out3)))
        # out3 += res
        # x = self.pool(out3)
        
        # x = F.relu(self.bn11(self.conv11(x)))
        # x = self.pool(x)
        
        # x = x.view(x.size()[0], 16 * 12 * 12)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x
