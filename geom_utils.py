import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import utils

def compute_jacobian(f, z, h=1e-5):
  fz = f(z).squeeze()
  z = z.squeeze()
  n = fz.size()
  d = z.size()
  J = torch.zeros(tuple(n) + tuple(d))
  #for i in range(d):
  #  zh = z.copy()
  #  zh[i] += h
  #  fzh = f(zh).squeeze()
  #  J[:, i] = (fzh - fz) / h
  
  return J

def show_img(currx):
  utils.save_image(
    currx,
    f"test.png",
    nrow=1,
    normalize=True,
    range=(-1, 1),
  )
  plt.imshow(cv2.imread('test.png')[:, :, ::-1])
  plt.show()

def load_obama():
  im = cv2.imread("obama.jpg")
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  im = np.swapaxes(im,1,2)
  im = np.swapaxes(im,0,1)
  im = torch.from_numpy(im)
  im = torch.unsqueeze(im,0)
  im = im.float()
  im = 2 * im / 255
  im = im - 1
  return im