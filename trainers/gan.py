import torch
from torch import nn, optim
from torch.utils import data
from torchvision.utils import save_image

from metrics import Average


class Trainer(object):

    def __init__(self, net_g: nn.Module, net_d: nn.Module,
                 optim_g: optim.Optimizer, optim_d: optim.Optimizer,
                 dataloader: data.DataLoader, device: torch.device,
                 num_d_iterations: int):
        self.net_g = net_g
        self.net_d = net_d
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.dataloader = dataloader
        self.device = device

        self.loss_d = Average()
        self.loss_g = Average()

        self.num_d_iterations = num_d_iterations

    def train_epoch(self):
        self.loss_d.reset()
        self.loss_g.reset()

        for i, x in enumerate(self.dataloader):
            x = x.to(self.device)
            z = torch.randn(x.size(0), 128, dtype=torch.float).to(self.device)
            if (i + 1) % (self.num_d_iterations + 1) == 0:
                # train discriminator
                fake = self.net_g(z).detach()
                loss_d = -torch.log(
                    torch.sigmoid(self.net_d(x) - self.net_d(fake))).mean()

                self.optim_d.zero_grad()
                loss_d.backward()
                self.optim_d.step()

                self.loss_d.update(loss_d.item(), x.size(0))
            else:
                # train generator
                fake = self.net_g(z)
                loss_g = -torch.log(
                    torch.sigmoid(self.net_d(fake) - self.net_d(x))).mean()

                self.optim_g.zero_grad()
                loss_g.backward()
                self.optim_g.step()

                self.loss_g.update(loss_g.item(), x.size(0))

    def train(self, epochs):
        for epoch in range(1, epochs):
            self.train_epoch()

            print(
                'Epoch: {}/{}, loss d: {loss_d.average:.6f}, loss g: {loss_g.average:.6f}.'.
                format(epoch, epochs, loss_d=self.loss_d, loss_g=self.loss_g))

            f = 'samples/{:2d}.jpg'.format(epoch)
            self.plot_sample(f)

            torch.save(self.net_d.state_dict(),
                       'models/net_d_{}.pt'.format(epoch))
            torch.save(self.net_g.state_dict(),
                       'models/net_g_{}.pt'.format(epoch))

    def plot_sample(self, f):
        z = torch.randn(64, 128, dtype=torch.float).to(self.device)
        with torch.no_grad():
            fake = self.net_g(z)
        save_image(fake.data, f, normalize=True, nrow=8)
