import sys
sys.path.append('stylegan2-pytorch')
sys.argv = ['-f']
from model import Generator
import torch
import numpy as np
import argparse
from torchvision import utils
from generate import generate
import torch.optim as optim

def load_generator():
	device = "cuda"

	parser = argparse.ArgumentParser(description="Generate samples from the generator")

	parser.add_argument(
	    "--size", type=int, default=1024, help="output image size of the generator"
	)
	parser.add_argument(
	    "--sample",
	    type=int,
	    default=1,
	    help="number of samples to be generated for each image",
	)
	parser.add_argument(
	    "--pics", type=int, default=20, help="number of images to be generated"
	)
	parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
	parser.add_argument(
	    "--truncation_mean",
	    type=int,
	    default=4096,
	    help="number of vectors to calculate mean for the truncation",
	)
	parser.add_argument(
	    "--ckpt",
	    type=str,
	    default="stylegan2-ffhq-config-f.pt",
	    help="path to the model checkpoint",
	)
	parser.add_argument(
	    "--channel_multiplier",
	    type=int,
	    default=2,
	    help="channel multiplier of the generator. config-f = 2, else = 1",
	)

	args = parser.parse_args()

	args.latent = 512
	args.n_mlp = 8

	g_ema = Generator(
	    args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
	).to(device)
	checkpoint = torch.load(args.ckpt)

	g_ema.load_state_dict(checkpoint["g_ema"])

	if args.truncation < 1:
	    with torch.no_grad():
	        mean_latent = g_ema.mean_latent(args.truncation_mean)
	else:
	    mean_latent = None

	return g_ema, args, mean_latent

def model_decode(model, latent, truncation, mean_latent, is_z=True):
  return model([latent], truncation=truncation, truncation_latent=mean_latent, input_is_latent=not is_z)[0]

def get_graddir(g_ema, z0,z1, truncation, mean_latent):
  x1 = model_decode(g_ema, z1, truncation, mean_latent).detach()
  optimizer = optim.Adam([z0], lr=0.001)
  z0.requires_grad = True
  x0 = model_decode(g_ema, z0, truncation, mean_latent)
  loss = torch.mean((x0 - x1)**2)
  loss.backward()
  grad_ret = torch.clone(z0.grad.detach())
  g_ema.zero_grad()
  return grad_ret

def get_loss(g_ema, z0,z1, truncation, mean_latent):
  x1 = model_decode(g_ema, z1, truncation, mean_latent).detach()
  optimizer = optim.Adam([z0], lr=0.001)
  z0.requires_grad = True
  x0 = model_decode(g_ema, z0, truncation, mean_latent).detach()
  with torch.no_grad():
    loss = torch.mean((x0 - x1)**2)
    return loss







