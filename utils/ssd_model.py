import torch
import torch.nn as nn

def block(c_in, c_out, k=3, p=1, s=1, pk=2, ps=2):
    return torch.nn.Sequential(
        torch.nn.Conv2d(c_in, c_out, k, padding=p, stride=s),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(c_out),
        torch.nn.MaxPool2d(pk, stride=ps)
    )

def flatten_conv(x, k):
    return x.view(x.size(0), x.size(1) // k, -1).transpose(1, 2)

class out_conv(nn.Module):
    def __init__(self, c_in, k, n_classes):
        super().__init__()
        self.k = k
        self.oconv1 = nn.Conv2d(c_in, k * 4, 3, padding=1)
        self.oconv2 = nn.Conv2d(c_in, k * n_classes, 3, padding=1)

    def forward(self, x):
        return [
            flatten_conv(self.oconv1(x), self.k),
            flatten_conv(self.oconv2(x), self.k)
        ]

def conv(c_i, c_o, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(c_i, c_o, 3, stride=stride, padding=padding),
        nn.ReLU(),
        # nn.BatchNorm2d(c_o)
    )

class SSD(nn.Module):
    def __init__(self, n_channels=3, n_classes=20, k=[1, 1, 1]):
        super().__init__()
        # backbone
        self.conv1 = block(n_channels, 8)
        self.conv2 = block(8, 16)
        self.conv3 = block(16, 32)
        self.conv4 = block(32, 64)
        self.conv5 = block(64, 64)
        self.conv6 = block(64, 64)
        # head
        self.k = k
        self.out4 = out_conv(64, self.k[0], n_classes)
        self.out5 = out_conv(64, self.k[1], n_classes)
        self.out6 = out_conv(64, self.k[2], n_classes)

    def forward(self, x):
        # backbone
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # para inputs de 100x100
        x4 = self.conv4(x)  # 6x6
        x5 = self.conv5(x4)  # 3x3
        x6 = self.conv6(x5)  # 1x1

        # head
        o1l, o1c = self.out4(x4)
        o2l, o2c = self.out5(x5)
        o3l, o3c = self.out6(x6)
        return torch.cat([o1l, o2l, o3l], dim=1), torch.cat([o1c, o2c, o3c], dim=1)

if __name__ == "__main__":
    classes = ["background", "agave"]  # Incluye background y agave
    n_classes = len(classes)  # Esto debería ser 2
    k = [3, 3, 3]  # Example anchor ratios
    net = SSD(n_classes=n_classes, k=k)  # Usa n_classes correcto
    output = net(torch.rand((64, 3, 100, 100)))
    print(output[0].shape, output[1].shape)
