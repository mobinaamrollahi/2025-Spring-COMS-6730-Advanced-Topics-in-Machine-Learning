import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        ### YOUR CODE HERE
        # print("in the network.py.")
        self.version = args.resnet_version
        self.num_blocks = args.resnet_size
        self.num_classes = args.num_classes
        self.drop_rate = args.drop
        self.batch = args.batch
        
        # Flags to use batch normalization and drop out
        self.use_bn = args.use_bn
        self.use_dropout = args.use_dropout

        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.in_channels) if self.use_bn else nn.Identity()
        self.relu = nn.ReLU()

        block = BasicBlock if self.version == 1 else BottleneckBlock

        self.layer1 = self._make_layer(block, 16, self.num_blocks, stride=1)
        self.layer2 = self._make_layer(block, 32, self.num_blocks, stride=2)
        self.layer3 = self._make_layer(block, 64, self.num_blocks, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, self.num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(
                block(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=s,
                    use_bn=self.use_bn,
                    use_dropout=self.use_dropout,
                    drop_rate=self.drop_rate
                )
            )
            self.in_channels = out_channels
        return nn.Sequential(*layers)
        ### END YOUR CODE

    def forward(self, x):
        '''
        Input x: a batch of images (batch size x 3 x 32 x 32)
        Return the predictions of each image (batch size x 10)
        '''
        ### YOUR CODE HERE
        # print("in the network.py.")
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1) 
        out = self.fc(out)
        ### END YOUR CODE
        return out
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_bn, use_dropout, drop_rate):
        super().__init__()
        self.same_shape = (in_channels == out_channels and stride == 1)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.dropout = nn.Dropout(drop_rate) if use_dropout else nn.Identity()

        if not self.same_shape:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)
    
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_bn, use_dropout, drop_rate):
        super().__init__()
        mid_channels = out_channels // 4
        self.same_shape = (in_channels == out_channels and stride == 1)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(in_channels) if use_bn else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(mid_channels) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(mid_channels) if use_bn else nn.Identity()
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)

        self.dropout = nn.Dropout(drop_rate) if use_dropout else nn.Identity()

        if not self.same_shape:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.dropout(out)
        out = self.conv3(self.relu(self.bn3(out)))
        return out + self.shortcut(x)



