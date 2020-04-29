import torch.nn as nn


# generator G(z)
class Generator(nn.Module):
    def __init__(self, zdim=100, d=64):
        super(Generator, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(zdim, d*8, 2, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.deconv1_bn(self.deconv1(x)))
        x = self.relu(self.deconv2_bn(self.deconv2(x)))
        x = self.relu(self.deconv3_bn(self.deconv3(x)))
        x = self.relu(self.deconv4_bn(self.deconv4(x)))
        x = self.tanh(self.deconv5(x))
        return x


# discriminator
class Discriminator(nn.Module):
    def __init__(self, d=64, classes=10, leaky_k=0.2):
        super(Discriminator, self).__init__()

        self.leaky_k = leaky_k
        self.leaky_relu = nn.LeakyReLU(leaky_k, inplace=True)

        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5_dis = nn.Conv2d(d*8, 1, 2, 1, 0)
        self.conv5_aux = nn.Conv2d(d*8, classes, 2, 1, 0)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2_bn(self.conv2(x)))
        x = self.leaky_relu(self.conv3_bn(self.conv3(x)))
        x = self.leaky_relu(self.conv4_bn(self.conv4(x)))
        dis = self.conv5_dis(x).squeeze()
        aux = self.conv5_aux(x).squeeze()
        return dis, aux
