import mindspore.nn as nn


class ResBlockDS(nn.Cell):
#在这里进行补全！！
    def __init__(self, in_channel, out_channel):
        super(ResBlockDS, self).__init__()
        self.seq = nn.SequentialCell(
            [
                nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1, pad_mode='pad'),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, pad_mode='pad'),
                nn.BatchNorm2d(out_channel),
            ])
        self.relu = nn.ReLU()
        
        self.shortup = nn.SequentialCell(
            [
                nn.Conv2d(in_channel, out_channel, 1, stride=2),
                nn.BatchNorm2d(out_channel),
            ])
    def construct(self, x):
        y = self.seq(x) + self.shortup(x)
        #先relu后maxpooling
        y = self.relu(y)
        return y

class ResBlock(nn.Cell):
#在这里进行补全！！
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.seq = nn.SequentialCell(
            [
                nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1, pad_mode='pad'),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, pad_mode='pad'),
                nn.BatchNorm2d(out_channel),
            ])
        self.relu = nn.ReLU()

    def construct(self, x):
        y = self.seq(x)
       
        y = self.relu(y)
        return y

class ResNet18(nn.Cell):
    def __init__(self, class_num=10, in_channel=3):
        super(ResNet18, self).__init__()
        self.conv = nn.Conv2d(in_channel, 64, 7, stride=2, padding=3, pad_mode='pad')
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.ResBlock1 = ResBlock(64, 64)
        self.ResBlock2 = ResBlock(64, 64)
        
        self.ResBlockDS3 = ResBlockDS(64, 128)
        
        self.ResBlock4 = ResBlock(128, 128)
        self.ResBlockDS5 = ResBlockDS(128, 256)
        
        self.ResBlock6 = ResBlock(256, 256)
        self.ResBlockDS7 = ResBlockDS(256, 512)
        self.ResBlock8 = ResBlock(512, 512)
        self.LastProcess = nn.SequentialCell([
            nn.AvgPool2d(7),
            nn.Flatten(),
            nn.Dense(512, class_num),
        ])

    def construct(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.ResBlock1(x)
        x = self.ResBlock2(x)
        x = self.ResBlockDS3(x)
        x = self.ResBlock4(x)
        x = self.ResBlockDS5(x)
        x = self.ResBlock6(x)
        x = self.ResBlockDS7(x)
        x = self.ResBlock8(x)
        y = self.LastProcess(x)
        return y