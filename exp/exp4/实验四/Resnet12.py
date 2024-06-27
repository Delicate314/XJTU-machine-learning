import mindspore.nn as nn


class ResBlock(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.seq = nn.SequentialCell(
            [
                nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1, pad_mode='pad'),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, pad_mode='pad'),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, pad_mode='pad'),
                nn.BatchNorm2d(out_channel),
            ])
        self.shortup = nn.SequentialCell(
            [
                nn.Conv2d(in_channel, out_channel, 1, stride=1),
                nn.BatchNorm2d(out_channel),
            ])
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def construct(self, x):
        y = self.seq(x) + self.shortup(x)
        y = self.relu(y)
        return self.max_pool(y)


class ResNet12(nn.Cell):
    def __init__(self, class_num=10, input_size=224):
        super(ResNet12, self).__init__()
        self.ResBlock1 = ResBlock(3, 64)
        self.ResBlock2 = ResBlock(64, 128)
        self.ResBlock3 = ResBlock(128, 256)
        self.ResBlock4 = ResBlock(256, 512)
        self.LastProcess = nn.SequentialCell([
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Dense(512, 1024),
            nn.ReLU(),
            nn.Dense(1024, 512),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Dense(512, class_num),
        ])

    def construct(self, x):
        x = self.ResBlock1(x)
        x = self.ResBlock2(x)
        x = self.ResBlock3(x)
        x = self.ResBlock4(x)
        y = self.LastProcess(x)
        return y