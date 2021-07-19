import torch.nn.functional as F
import torch.optim as optim

def minimize_path_energy(model, z0, z1, numpts=10,numtrainiter=1000,learning_rate=0.001,verbose=False):
  """
  Initialization: a linear interpolating path between z0 and z1 with numpts number of points.
  Output: a path with numpts between z0 and z1 trained to minimize energy with respect to the model.
  """

  # Initialize the interpolation between z0 and z1 with a linear interpolation.
  t = torch.linspace(1 / (numpts + 1), numpts / (numpts + 1), numpts).to(device)
  interp_points = torch.outer(1 - t, torch.flatten(z0)) + torch.outer(t, torch.flatten(z1))
  interp_points.requires_grad = True

  # For convenience, precompute x0 and x1.
  x0 = model.decode(z0).detach()
  x1 = model.decode(z1).detach()

  optimizer = optim.Adam([interp_points], lr=learning_rate)
  for i in range(numtrainiter):
    imgs = model.decode(interp_points)

    # compute the energy of the path
    loss = torch.sum((torch.cat((x0, imgs)) - torch.cat((imgs, x1)))**2)

    # # compute the length of the path
    # loss = torch.sqrt(torch.sum((torch.cat((x0, imgs)) - torch.cat((imgs, x1)))**2))

    if verbose:
      with torch.no_grad():
        if i % 1000 == 0:
          print('loss',i,loss.to('cpu'))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  imgs = model.decode(interp_points)

  # compute the energy of the path
  loss = torch.sum((torch.cat((x0, imgs)) - torch.cat((imgs, x1)))**2)

  # # compute the length of the path
  # loss = torch.sqrt(torch.sum((torch.cat((x0, imgs)) - torch.cat((imgs, x1)))**2))

  return torch.cat((z0, interp_points.detach(), z1)), loss.detach()

def get_curve_length(model, zseq):
  length = 0
  for i in range(zseq.shape[0]-1):
    length = length + torch.sqrt(torch.sum((model.decode(zseq[i,:]) - model.decode(zseq[i+1,:])) ** 2))
  return length

def minimize_straight_line_energy(model, z0, z1, numpts=10,numtrainiter=1000,learning_rate=0.001,verbose=False):
  """
  Initialization: a linear interpolating path between z0 and z1 with numpts number of points.
  Output: a path with numpts between z0 and z1 trained to minimize energy with respect to the model.
  """

  # Initialize the interpolation between z0 and z1 with a linear interpolation.
  t = torch.linspace(1 / (numpts + 1), numpts / (numpts + 1), numpts).to(device)
  t.requires_grad = True

  # For convenience, precompute x0 and x1.
  x0 = model.decode(z0).detach()
  x1 = model.decode(z1).detach()

  optimizer = optim.Adam([t], lr=learning_rate)
  for i in range(numtrainiter):
    interp_points = torch.outer(1 - t, torch.flatten(z0)) + torch.outer(t, torch.flatten(z1))
    imgs = model.decode(interp_points)

    # compute the energy of the path
    loss = torch.sum((torch.cat((x0, imgs)) - torch.cat((imgs, x1)))**2)

    if verbose:
      with torch.no_grad():
        if i % 1000 == 0:
          print('loss',i,loss.to('cpu'))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  interp_points = torch.outer(1 - t, torch.flatten(z0)) + torch.outer(t, torch.flatten(z1))
  imgs = model.decode(interp_points)
  loss = torch.sum((torch.cat((x0, imgs)) - torch.cat((imgs, x1)))**2)
  return torch.cat((z0, interp_points.detach(), z1)), loss.detach()

def compute_path_length(model,path):
  l = 0
  for i in range(path.shape[0]-1):
    xip1 = model.decode(path[i+1,:])
    xi = model.decode(path[i,:])
    l += torch.norm(xip1 - xi)
  return l

