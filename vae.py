import os
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pickle
import sys
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import math
from models import VAE

def get_trained_vae(notebook_data_path, dataset_name,latent_d):
  model_path = notebook_data_path + 'trained_models/'
  if not os.path.exists(model_path):
    os.mkdir(model_path)
  vae_filename = model_path + dataset_name + '_vae_d' + str(latent_d) + '.pkl'
  if not os.path.exists(vae_filename):
    print('VAE does not exist. Training it now.')
    train_vae(notebook_data_path, dataset_name, latent_d, vae_filename)
  model = pickle.load(open(vae_filename, 'rb'))
  return model

 #@title Code: VAE training
"""
The following code is a slightly modified version of the pytorch library's
example directory for training a simple VAE.
"""
def train_vae(notebook_data_path, datasetname,latent_d,filename):
    sys.argv = ['-f']

    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--vis-interval',type=int, default=10, metavar='N',
                        help='how many batches to wait before dumping visualization')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


    dataset_name = datasetname
    if dataset_name == 'mnist':
        traindataset = datasets.MNIST(notebook_data_path + 'data/', train=True, download=True,
                      transform=transforms.ToTensor())

        testdataset = datasets.MNIST(notebook_data_path + 'data/', train=False, transform=transforms.ToTensor())

    elif dataset_name in ['angle', 'circleangle']:
        transformlist = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor()])
        traindataset = datasets.ImageFolder('./data/' + dataset_name + '/', transform=transformlist)
        testdataset = datasets.ImageFolder('./data/' + dataset_name + '/', transform=transformlist)

    elif dataset_name == 'untrained':
        args.epochs = 0
        # dummy code
        traindataset = datasets.MNIST(notebook_data_path + 'data/', train=True, download=True,
              transform=transforms.ToTensor())

        testdataset = datasets.MNIST(notebook_data_path + 'data/', train=False, transform=transforms.ToTensor())


    train_loader = torch.utils.data.DataLoader(traindataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testdataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def vae_loss_function(recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def vae_train(model, optimizer, epoch):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))
            # if batch_idx % args.vis_interval == 0:
                # vis_weights(model)

        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    model = VAE(latent_d = latent_d).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, args.epochs + 1):
        vae_train(model, optimizer, epoch)
    pickle.dump(model, open(filename, 'wb'))
    print('trained',latent_d)

