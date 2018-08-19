import argparse
import os

import torch
from torch import optim
from torchvision.utils import save_image

import loader
import model
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--data-parallel', action='store_true')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    device = torch.device('cuda' if args.cuda else 'cpu')

    net_g = model.Generator().to(device)
    net_d = model.Discriminator().to(device)

    optim_g = optim.Adam(
        net_g.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optim_d = optim.Adam(
        net_d.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    dataloader = loader.get_dataloader()

    trainer = Trainer(net_g, net_d, optim_g, optim_d, dataloader, device)

    os.makedirs('samples', exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        loss_d_avg, loss_g_avg = trainer.train()
        print('Epoch: {}/{}, loss d: {:.6f}, loss g: {:.6f}.'.format(
            epoch, args.epochs, loss_d_avg, loss_g_avg))

        z = torch.randn(64, 128, dtype=torch.float).to(device)
        fake = trainer.net_g(z)

        save_image(
            fake.data,
            'samples/{:2d}.jpg'.format(epoch),
            normalize=True,
            nrow=8)


if __name__ == '__main__':
    main()
