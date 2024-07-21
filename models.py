import torch.nn as nn
import torch

def build_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class Rib(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels * 3
        self.left = build_block(in_channels, out_channels)
        self.right = build_block(in_channels, out_channels)
        self.mid = build_block(mid_channels, out_channels)

    def forward(self, left_x, right_x, mid_x=None):
        if mid_x is None:
            mid_x = torch.cat([left_x, right_x], dim=1)
        else:
            mid_x = torch.cat([mid_x, left_x, right_x], dim=1)

        return self.left(left_x), self.mid(mid_x), self.right(right_x)


class RibCage(nn.Module):
    def __init__(self, *args, channels, **kwargs):
        super().__init__(*args, **kwargs)
        self.rib1 = Rib(channels[0], channels[1], mid_channels=channels[0] * 2)
        self.rib2 = Rib(channels[1], channels[2])
        self.rib3 = Rib(channels[2], channels[3])
        self.rib4 = Rib(channels[3], channels[4])

        self.fc1 = nn.Linear(channels[4] * 3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, left, right):
        left, mid, right = self.rib1(left, right)
        left, mid, right = self.rib2(left, right, mid)
        left, mid, right = self.rib3(left, right, mid)
        left, mid, right = self.rib4(left, right, mid)
        bs = left.shape[0]

        left, mid, right = [self.avgpool(t) for t in [left, mid, right]]
        x = torch.cat([left.view(bs, -1), mid.view(bs, -1), right.view(bs, -1)], dim=1)
        x = self.fc1(x).relu()
        return self.fc3(self.fc2(x).relu())


class Naive(nn.Module):
    def __init__(self, channels, is_fc=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block1 = build_block(channels[0], channels[1])
        self.block2 = build_block(channels[1], channels[2])
        self.block3 = build_block(channels[2], channels[3])
        self.block4 = build_block(channels[3], channels[4])
        self.is_fc = is_fc
        if is_fc:
            self.fc1 = nn.Linear(channels[4], 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, left, right=None):
        if right is not None:
            x = torch.cat([left, right], dim=1)
        else:
            x = left
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x).view(x.shape[0], -1)
        if self.is_fc:
            x = self.fc1(x).relu()
            return self.fc3(self.fc2(x).relu())
        return x



class Siamese(nn.Module):
    def __init__(self, channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network1 = Naive(channels, is_fc=False)
        self.network2 = Naive(channels, is_fc=False)
        self.fc1 = nn.Linear(channels[-1] * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, left, right):
        x1 = self.network1(left)
        x2 = self.network2(right)
        x = torch.cat([x1, x2], dim=1)
        x = self.fc1(x).relu()
        return self.fc3(self.fc2(x).relu())


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    rib_channels = [1, 32, 64, 128, 256]
    rib_model = RibCage(channels=rib_channels)

    naive_channels = [int(t * 2) for t in [1, 32, 64, 128, 256]]
    naive_model = Naive(channels=naive_channels)

    siamese_channels = [1] + [int(t * 1.6) for t in [32, 64, 128, 256]]
    siamese_model = Siamese(channels=siamese_channels)


    # sanity check
    # print(rib_model(torch.randn(1, 1, 256, 256), torch.randn(1, 1, 256, 256)).shape)
    # print(naive_model(torch.randn(1, 2, 256, 256)).shape)
    # print(siamese_model(torch.randn(1, 1, 256, 256), torch.randn(1, 1, 256, 256)).shape)
    #
    #
    # for model, name in zip([rib_model, naive_model, siamese_model], ['RibCage', 'Naive Model', 'Siamese Model']):
    #     total_params = count_parameters(model)
    #     print(f"Total trainable parameters in {name}: {total_params}")


    print(siamese_model)