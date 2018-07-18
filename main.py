import argparse

import torch
from torch import optim

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

    net_g = model.Generator()
    net_d = model.Discriminator()

    optim_g = optim.Adam(
        net_g.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optim_d = optim.Adam(
        net_d.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    dataloader = loader.get_dataloader()

    device = torch.device('gpu' if args.cuda else 'cpu')

    trainer = Trainer(net_d, net_g, optim_d, optim_g, dataloader, device)

    for epoch in range(args.epochs):
        trainer.train()


if __name__ == '__main__':
    main()
