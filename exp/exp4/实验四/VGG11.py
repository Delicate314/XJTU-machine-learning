import mindspore.nn as nn


class VGG11(nn.Cell):
    """
    网络结构
    """
    '''
    任务一补全
    '''
    def __init__(self, num_class=10, num_channel=3):
        super(VGG11,self).__init__()
        self.conv1 = self.conv_block(num_channel, 64)
        self.conv2 = self.conv_block(64, 128)
        self.conv3 = self.conv_block(128, 256)
        self.conv4 = self.conv_block(256,256)
        self.conv5 = self.conv_block(256, 512)
        self.conv6 = self.conv_block(512, 512)
        self.conv7 = self.conv_block(512, 512)
        self.conv8 = self.conv_block(512, 512)
        self.fc9 = nn.Dense(25088, 6272)
        self.fc10 = nn.Dense(6272, 1568)
        self.fc11 = nn.Dense(1568, num_class)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_block(self, in_channel, out_channel):
        seq = nn.SequentialCell(
            [
                nn.Conv2d(in_channel, out_channel, 3, padding=1, pad_mode='pad'),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            ])
        return seq

    def construct(self, x):
        # 使用定义好的运算构建前向网络
        x = self.conv1(x)
        x = self.max_pool2d(x)

        x = self.conv2(x)
        x = self.max_pool2d(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool2d(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max_pool2d(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.max_pool2d(x)
        
        x = self.flatten(x)

        x = self.fc9(x)
        x = self.relu(x)
        x = self.fc10(x)
        x = self.relu(x)
        x = self.fc11(x)
        
        return x