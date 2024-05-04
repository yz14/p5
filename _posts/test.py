""" RNN models, LSTM, GRU """

import torch
import torch.nn as nn
from torch.optim import Adam

class GRU(nn.Module):
    def __init__(self, d_in, d_hid, n_layer, n_class):
        super(GRU, self).__init__()
        self.enc = nn.GRU(d_in, d_hid, n_layer, batch_first=True)
        self.dec = nn.Linear(d_hid, n_class)
    
    def forward(self, x, h0):
        hs, h = self.enc(x, h0)  # (bsz, seq_len, d_hid)
        logits = self.dec(hs[:, -1, :])
        return logits

if __name__ == "__main__":
    # config
    d_in, d_hid, n_layer, n_class = 28, 4, 2, 10
    bsz, lr = 2, 1e-4
    # data
    x = torch.randn(bsz, 1, 28, 28) # img
    y = torch.Tensor([1, 2]).long()
    # model
    model = GRU(d_in, d_hid, n_layer, n_class)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    # training
    h0 = torch.zeros(n_layer, bsz, d_hid)
    optimizer.zero_grad()
    logits = model(x.view(2, 28, 28), h0)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()
    print(loss.item())