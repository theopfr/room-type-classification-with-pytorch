
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, dropout_chance=0.0):
        super(Model, self).__init__()

        self.pretrained_model = torchvision.models.resnet18(pretrained=True, progress=True)
        self.pretrained_model.fc = torch.nn.Linear(512, 400)

        self.dense1 = nn.Linear(400, 250)
        self.dense2 = nn.Linear(250, 75)
        self.dense3 = nn.Linear(75, 4)

        self.dropout = nn.Dropout2d(p=dropout_chance)

    def forward(self, x, targets=None):
        # dense layer of the pretrained network
        x = F.relu(self.pretrained_model(x))
        
        # additional dense layers
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = F.relu(self.dense2(x))
        x = self.dropout(x)
        x = F.softmax(self.dense3(x), dim=1)

        return x
