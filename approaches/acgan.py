import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from networks.acgan_net import Generator, Discriminator
from utils.util import *


class Model(object):
    def __init__(self, nc=10, img_size=32, zdim=100, saved_dir=os.path.join('model', 'saved', 'saved')):
        super(Model, self).__init__()
        self.dir_g = saved_dir + '_g.pth'
        self.dir_d = saved_dir + '_d.pth'

        self.nc = nc
        self.zdim = zdim
        self.img_size = img_size

        self.l_w = 10.
        self.aux_k = 1000.

        self.G = Generator(zdim=zdim).cuda()
        self.D = Discriminator().cuda()
        print('model created successfully!')

        self.data_loader = None

        # loading
        if os.path.exists(self.dir_g) and os.path.exists(self.dir_d):
            self.G.load_state_dict(torch.load(self.dir_g))
            self.D.load_state_dict(torch.load(self.dir_d))
            print('model loaded!')
        else:
            self.G.apply(weights_init)
            self.D.apply(weights_init)
            print('model initialized!')

    def train(self, niter=3, batch_size=128, lr=1e-3):
        # data
        if self.data_loader is None:
            self.data_loader = train_loader(self.img_size, batch_size)
        data_loader = self.data_loader

        # loss function and optimizers

        CELoss = nn.CrossEntropyLoss()
        G_optimizer = optim.Adam(self.G.parameters(), lr, betas=(0.5, 0.999))
        D_optimizer = optim.Adam(self.D.parameters(), lr, betas=(0.5, 0.999))

        print('let\'s go!')
        start_time = time.time()

        for epoch in range(niter):
            G_losses = []
            D_losses = []
            acc_real = []
            acc_fake = []
            epoch_start_time = time.time()

            self.D.train()
            self.G.train()

            for i, data in enumerate(data_loader):

                batch_size = data[0].shape[0]

                ##################################################
                # (1) update D : maximize log(D(x)) + log(1 - D(G(z))) + p log(p) - gradient_penalty
                ##################################################

                # train with real

                self.D.zero_grad()

                real_image, real_label = data

                input_ = real_image.cuda()
                dis_output, aux_output = self.D(input_)

                aux_label = real_label.cuda()
                loss_D_dis_real = dis_output.mean()
                loss_D_aux_real = CELoss(aux_output, aux_label) * self.aux_k
                loss_D_real = - loss_D_dis_real + loss_D_aux_real

                loss_D_real.backward()

                # calculate accuracy
                acc_real.append(calc_acc(aux_output, aux_label))

                # train with fake

                fake_label = np.random.randint(0, self.nc, (batch_size,)).astype(np.int64)
                z_ = make_noise(self.nc, self.zdim, fake_label).view(batch_size, self.zdim, 1, 1).cuda()
                fake_image = self.G(z_)

                input_ = fake_image.detach()
                dis_output, aux_output = self.D(input_)

                aux_label = torch.tensor(fake_label).cuda()
                loss_D_dis_fake = dis_output.mean()
                loss_D_aux_fake = CELoss(aux_output, aux_label) * self.aux_k
                loss_D_fake = loss_D_dis_fake + loss_D_aux_fake

                loss_D_fake.backward()

                # gradient penalty

                alpha = torch.randn(batch_size, 1)
                alpha = alpha.expand(batch_size, real_image.numel() // batch_size).contiguous()
                alpha = alpha.view(batch_size, 3, self.img_size, self.img_size).cuda()
                interpolate = alpha * real_image.cuda() + (1 - alpha) * fake_image
                interpolate = interpolate.requires_grad_(True)
                disc_interpolate, _ = self.D(interpolate)
                gradient = torch.autograd.grad(outputs=disc_interpolate, inputs=interpolate,
                                               grad_outputs=torch.ones(disc_interpolate.size()).cuda(),
                                               create_graph=True, only_inputs=True)[0]
                gradient = gradient.view(gradient.size(0), -1)
                gradient_penalty = ((gradient.norm(2, dim=1) - 1) ** 2).mean() * self.l_w

                gradient_penalty.backward()

                # step

                loss_D = loss_D_real + loss_D_fake + gradient_penalty
                D_optimizer.step()

                D_losses.append(loss_D.item())

                # calculate accuracy
                acc_fake.append(calc_acc(aux_output, aux_label))

                ##################################################
                # (2) update G : maximize log(D(G(z))) + p log(p)
                ##################################################

                self.G.zero_grad()

                fake_label = np.random.randint(0, self.nc, (batch_size,)).astype(np.int64)
                z_ = make_noise(self.nc, self.zdim, fake_label).view(batch_size, self.zdim, 1, 1).cuda()
                fake_image = self.G(z_)

                input_ = fake_image
                dis_output, aux_output = self.D(input_)

                aux_label = torch.tensor(fake_label).cuda()
                loss_G_dis = dis_output.mean()
                loss_G_aux = CELoss(aux_output, aux_label) * self.aux_k

                loss_G = - loss_G_dis + loss_G_aux
                loss_G.backward()
                G_optimizer.step()

                G_losses.append(loss_G.item())

                if i % 40 == 0 and i != 0:
                    print('%d out of %d in epoch %d, D_loss %.4f, G_loss %.4f, acc %.2f %.2f' %
                          (i, len(data_loader), epoch, np_mean(D_losses[i-20:i]), np_mean(G_losses[i-20:i]),
                           np_mean(acc_real[i-20:i]), np_mean(acc_fake[i-20:i])))

            torch.save(self.G.state_dict(), self.dir_g)
            torch.save(self.D.state_dict(), self.dir_d)

            epoch_time = time.time() - epoch_start_time
            print('*' * 30)
            print('epoch %s finished' % epoch)
            print('*' * 30)
            print('[%d/%d] - time: %.2f, D_loss: %.3f, G_loss: %.3f, acc %.2f %.2f' %
                  (epoch, niter, epoch_time, np_mean(D_losses), np_mean(G_losses), np_mean(acc_real), np_mean(acc_fake)))
            print('*' * 30)
            
            self.do_generate(epoch)

        total_time = time.time() - start_time
        print("Avg per epoch time: %.2f, total %d epochs time: %.2f" % (total_time/niter, niter, total_time))
        print("Training finished!... save training results")

        torch.save(self.G.state_dict(), self.dir_g)
        torch.save(self.D.state_dict(), self.dir_d)
        
    def do_generate(self, id):
        print('...')
        label = []
        for i in range(10):
            for _ in range(10):
                label.append(i)
        self.generate(label, os.path.join('results', 'test', 'test_%02d' % id))

    def generate(self, label, file_path):
        z_ = make_noise(self.nc, self.zdim, np.array(label)).view(len(label), self.zdim, 1, 1).cuda()
        self.G.eval()
        imgs = self.G(z_)

        for i, img in enumerate(imgs):
            img = ((img.detach().cpu().numpy()+1)*127.5).astype(np.uint8).transpose((1, 2, 0))
            img = Image.fromarray(img, mode='RGB')
            img.save(file_path + '_%s' % i + '_%s' % label[i] + '.jpg')
