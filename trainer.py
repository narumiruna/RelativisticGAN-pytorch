import torch

from utils import AverageMeter


class Trainer(object):

    def __init__(self, net_g, net_d, optim_g, optim_d, dataloader, device):
        self.net_g = net_g
        self.net_d = net_d
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.dataloader = dataloader
        self.device = device

    def train(self):
        loss_d_avg = AverageMeter()
        loss_g_avg = AverageMeter()
        for x, _ in self.dataloader:
            # train discriminator

            x = x.to(self.device)
            z = torch.randn(x.size(0), 128, dtype=torch.float).to(self.device)

            fake = self.net_g(z).detach()
            loss_d = -torch.log(
                torch.sigmoid(self.net_d(x) - self.net_d(fake))).mean()

            self.optim_d.zero_grad()
            loss_d.backward()
            self.optim_d.step()

            # train generator
            z = torch.randn(x.size(0), 128, dtype=torch.float).to(self.device)

            fake = self.net_g(z)
            loss_g = -torch.log(
                torch.sigmoid(self.net_d(fake) - self.net_d(x))).mean()

            self.optim_g.zero_grad()
            loss_g.backward()
            self.optim_g.step()

            loss_d_avg.update(loss_d.item(), x.size(0))
            loss_g_avg.update(loss_g.item(), x.size(0))

        return loss_d_avg.average, loss_g_avg.average
