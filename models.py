from torch import nn, optim
from torch.nn import functional as F

"""
The following code is a slightly modified version of the pytorch library's
example directory for representing a simple VAE.
"""
class VAE(nn.Module):
    def __init__(self, latent_d=20):
        super(VAE, self).__init__()

        self.latent_d = latent_d
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_d)
        self.fc22 = nn.Linear(400, latent_d)
        self.fc3 = nn.Linear(latent_d, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar