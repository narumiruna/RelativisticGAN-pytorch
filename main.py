import argparse
import os

import torch
from torch import optim

from datasets import get_cat_dataloader
from networks import Discriminator, Generator
from trainers import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--data-parallel', action='store_true')
    parser.add_argument('--num-d-iterations', type=int, default=1)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    device = torch.device('cuda' if args.cuda else 'cpu')

    net_g = Generator(ch=128).to(device)
    net_d = Discriminator(ch=128).to(device)

    optim_g = optim.Adam(
        net_g.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optim_d = optim.Adam(
        net_d.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    dataloader = get_cat_dataloader()

    trainer = Trainer(net_g, net_d, optim_g, optim_d, dataloader, device,
                      args.num_d_iterations)

    os.makedirs('samples', exist_ok=True)

    trainer.train(args.epochs)


if __name__ == '__main__':
    main()
