import torch
import torch.nn as nn
import math


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        # SE
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(
            planes * 4, planes // 4, kernel_size=1, bias=False)
        self.conv_up = nn.Conv2d(
            planes // 4, planes * 4, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()
        # Downsample
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out1 = self.global_pool(out)
        out1 = self.conv_down(out1)
        out1 = self.relu(out1)
        out1 = self.conv_up(out1)
        out1 = self.sig(out1)

        if self.downsample is not None:
            residual = self.downsample(x)

        res = out1 * out + residual
        res = self.relu(res)

        return res


class SEResNet(nn.Module):

    def __init__(self, block, layers, num_classes=4000):
        self.inplanes = 64
        super(SEResNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) #原始
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        # self.layer2 = self._make_layer(block, 64, layers[1], stride=2)  # 原始
        # self.layer3 = self._make_layer(block, 128, layers[2], stride=2)  # 原始
        # self.layer4 = self._make_layer(block, 256, layers[3], stride=2)  # 原始
        # self.layer2 = self._make_layer(block, 64, layers[1], stride=[2, 1])   #这里stride=2改为stride=[2, 1]
        # self.layer3 = self._make_layer(block, 128, layers[2], stride=[2, 1])   #这里stride=2改为stride=[2, 1]
        # self.layer4 = self._make_layer(block, 256, layers[3], stride=[2, 1])   #这里stride=2改为stride=[2, 1]
        self.layer2 = self._make_layer(block, 64, layers[1], stride=1)   #这里stride=2改为stride=1
        self.layer3 = self._make_layer(block, 128, layers[2], stride=1)   #这里stride=2改为stride=1
        self.layer4 = self._make_layer(block, 256, layers[3], stride=1)   #这里stride=2改为stride=1
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.size())

        x = self.maxpool(x)
        # print("first maxpool:", x.size())
        x = self.layer1(x)
        # x2 = x
        # print("first layer:", x.size())
        x = self.layer2(x)
        # print("second layer:", x.size())
        x = self.layer3(x)
        # print("third layer:", x.size())
        x = self.layer4(x)
        # print("fourth layer:", x.size())
        # x = self.avgpool(x)
        # print("avgpool layer:", x.size())
        # x = x.view(x.size(0), -1)
        # print("view:", x.size())
        # x = self.fc(x)
        return x

def se_resnet50():
    model = SEResNet(Bottleneck, [3, 4, 6, 3])
    # print(model)
    return model


def test():
    x = torch.randn(64,1,32,100)
    model = se_resnet50()
    y = model(x)
    print(y.size())

if __name__ == '__main__':
    test()