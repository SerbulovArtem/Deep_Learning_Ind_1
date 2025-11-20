import torch
from torch import nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=(5,5), padding=2, stride=1),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.LazyConv2d(16, kernel_size=(5, 5), padding=0, stride=1),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.Flatten(),
            nn.LazyLinear(120),
            nn.Sigmoid(),
            nn.LazyLinear(84),
            nn.Sigmoid(),
            nn.LazyLinear(num_classes)
        )

    def forward(self, x, y=None):
        logits = self.net(x)
        if y is None:
            return logits
        loss = self.criterion(logits, y)
        return logits, loss
    

class AlexNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

        self.net = nn.Sequential(
            nn.LazyConv2d(96, kernel_size=(11, 11), stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.LazyConv2d(256, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.LazyConv2d(384, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.LazyConv2d(384, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.LazyConv2d(256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Flatten(),
            nn.LazyLinear(4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.LazyLinear(4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.LazyLinear(num_classes)
        )

    def forward(self, x, y=None):
        logits = self.net(x)
        if y is None:
            return logits
        loss = self.criterion(logits, y)
        return logits, loss
    

class VGGBlock(nn.Module):
    def __init__(self, num_conv_output, num_conv_layers):
        super().__init__()
        module_list = nn.ModuleList()
        for _ in range(num_conv_layers):
            module_list.append(nn.LazyConv2d(num_conv_output, kernel_size=(3, 3), padding=1))
            module_list.append(nn.ReLU())
        module_list.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.net = nn.Sequential(*module_list)
    
    def forward(self, x):
        return self.net(x)


class VGGNet(nn.Module):
    def __init__(self, config, num_classes=100):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        module_list = nn.ModuleList()

        for num_conv_output, num_conv_layers in config:
            module_list.append(VGGBlock(num_conv_output, num_conv_layers))

        self.net = nn.Sequential(
            *module_list,
            nn.Flatten(),
            nn.LazyLinear(4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.LazyLinear(4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.LazyLinear(num_classes)
        )

    def forward(self, x, y=None):
        logits = self.net(x)
        if y is None:
            return logits
        loss = self.criterion(logits, y)
        return logits, loss
    

class NiNBlock(nn.Module):
    def __init__(self, num_conv_output, kernel_size, stride, padding):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(num_conv_output, kernel_size, stride, padding),
            nn.ReLU(),
            nn.LazyConv2d(num_conv_output, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.LazyConv2d(num_conv_output, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
    
    def forward(self, x):
        return self.net(x)


class NiN(nn.Module):
    def __init__(self, config, num_classes=100):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        module_list = nn.ModuleList()

        for num_conv_output, kernel_size, stride, padding in config:
            module_list.append(NiNBlock(num_conv_output, kernel_size, stride, padding))

        self.net = nn.Sequential(
            *module_list,
            nn.Dropout(0.5),
            NiNBlock(num_classes, kernel_size=(3, 3), stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x, y=None):
        logits = self.net(x)
        if y is None:
            return logits
        loss = self.criterion(logits, y)
        return logits, loss
    

class InceptionBlock(nn.Module):
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=(1, 1))

        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=(1, 1))
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=(3, 3), padding=1)

        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=(1, 1))
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=(5, 5), padding=2)

        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=(1, 1))

    def forward(self, x):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

        self.net = nn.Sequential(
            # Stem
            nn.LazyConv2d(64, kernel_size=(7, 7), stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            # Body
            nn.LazyConv2d(64, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.LazyConv2d(192, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            InceptionBlock(64, (96, 128), (16, 32), 32),
            InceptionBlock(128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            InceptionBlock(192, (96, 208), (16, 48), 64),
            InceptionBlock(160, (112, 224), (24, 64), 64),
            InceptionBlock(128, (128, 256), (24, 64), 64),
            InceptionBlock(112, (144, 288), (32, 64), 64),
            InceptionBlock(256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            InceptionBlock(256, (160, 320), (32, 128), 128),
            InceptionBlock(384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1,1)),
            # Head 
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        )

    def forward(self, x, y=None):
        logits = self.net(x)
        if y is None:
            return logits
        loss = self.criterion(logits, y)
        return logits, loss


class TimmWithLoss(nn.Module):
    def __init__(self, model, num_classes=100):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        logits = self.model(x)
        if y is None:
            return logits
        loss = self.criterion(logits, y)
        return logits, loss
    

class ViT(TimmWithLoss):
    def __init__(self, model, num_classes=100):
        super().__init__(model, num_classes)