""" RNN models, LSTM, GRU """

import torch
import torch.nn as nn
from torch.optim import Adam

class TextCNN(nn.Module):
    def __init__(self, vocab_size, n_class, d_emb, ch, seq_len, filter_sizes=[2,2,2]):
        super(TextCNN, self).__init__()
        d_hid = ch * len(filter_sizes)
        self.emb = nn.Embedding(vocab_size, d_emb)
        # [n_batch, 1, seq_len, d_emb] => [n_batch, ch, h, 1]
        self.cnns = nn.ModuleList([nn.Conv2d(1, ch, (size, d_emb)) for size in filter_sizes])
        # [n_batch, ch, h, 1] => [n_batch, ch, 1, 1]
        self.pools = [nn.MaxPool2d((seq_len-size+1, 1)) for size in filter_sizes] # d_out = d_in // k
        self.clf = nn.Linear(d_hid, n_class)

    def forward(self, x):
        n_batch, seq_len = x.shape
        x = self.emb(x) # [n_batch, seq_len, d_emb]
        x = x.unsqueeze(1) # [n_batch, 1, seq_len, d_emb]

        z = []
        for conv, pool in zip(self.cnns, self.pools):
            h = F.relu(conv(x)) # [n_batch, ch, seq_len, 1]
            h = pool(h) # [n_batch, ch, 1, 1]
            z.append(h)

        z = torch.cat(z, 1)
        z = z.view(n_batch, -1)
        z = self.clf(z)
        return z

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