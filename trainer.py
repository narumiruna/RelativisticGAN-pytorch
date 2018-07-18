import model
import torch


class Trainer(object):

    def __init__(self, net_g, net_d, optim_g, optim_d, dataloader, device):
        self.net_g = net_g
        self.net_d = net_d
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.dataloader = dataloader
        self.device = device

    def train(self):
        for x, _ in self.dataloader:
            z = torch.randn(x.size(0), 128, dtype=torch.float)

            fake = self.net_g(z)
