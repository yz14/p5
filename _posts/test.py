
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, d_in, d_hid, d_mu):
        super(VAE, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.ReLU())
        
        self.dec = nn.Sequential(
            nn.Linear(d_mu, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, d_in),
            nn.Sigmoid())
        
        self.mu = nn.Linear(d_hid, d_mu)
        self.var = nn.Linear(d_hid, d_mu)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.enc(x)
        mu, logvar = self.mu(x), self.var(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar
    
def loss_fn(x_rec, x, mu, logvar):
    BCE = F.binary_cross_entropy(x_rec, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

if __name__ == "__main__":
    # config
    d_in, d_hid, d_mu, lr = 784, 128, 16, 1e-4
    # data
    x = (torch.randn(2, 1, 28, 28) > 0) * 1.0 # img
    # model
    model = VAE(d_in, d_hid, d_mu)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    x_rec, mu, logvar = model(x)
    loss = loss_fn(x_rec, x, mu, logvar)
    print(loss.item())
