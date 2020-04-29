import torch.utils.data
from torchvision import datasets, transforms
import numpy as np


# initialization of network
def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif class_name.find('Embedding') != -1:
        m.weight.data.normal_(0.0, 0.02)


# calculate accuracy
def calc_acc(pred, label):
    pred = pred.max(1)[1]
    cnt = pred.eq(label).sum()
    return float(cnt) / float(len(label)) * 100


# data loader
def train_loader(img_size, batch_size):
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.Resize(img_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                              ])),
        batch_size=batch_size, shuffle=True)
    return loader


# latent noise
def make_noise(num_classes, zdim, label):
    batch_size = label.shape[0]
    noise = np.random.normal(0, 1, (batch_size, zdim)).astype(np.float32)
    label_onehot = np.zeros((batch_size, num_classes))
    label_onehot[np.arange(batch_size), label] = 1.
    noise[np.arange(batch_size), :num_classes] = label_onehot[np.arange(batch_size)]
    noise = torch.from_numpy(noise).view(batch_size, zdim)
    return noise


def np_mean(l):
    return np.array(l).mean()
