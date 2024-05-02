""" RNN models, LSTM, GRU """

import torch
import torch.nn as nn
from torch.optim import Adam


# class BiRNN(nn.Module):
#     def __init__(self, d_in, d_hid, n_layer, n_class):
#         super(BiRNN, self).__init__()
#         self.d_hid = d_hid
#         self.n_layer = n_layer
#         self.lstm = nn.LSTM(d_in, d_hid, n_layer, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(d_hid*2, n_class)  # 2 for bidirection
    
#     def forward(self, x, hc):
#         n_batch, seq_len, d_in = x.shape
#         # forward
#         out, _ = self.lstm(x, (h0, c0))  # (n_batch, seq_len, d_hid*2)
        
#         # use hidden of the last time step
#         out = self.fc(out[:, -1, :])
#         return out

# if __name__ == "__main__":
#     d_in, d_hid, n_layer, n_class = 28, 64, 2, 10
#     model = BiRNN(d_in, d_hid, n_layer, n_class)
#     x = torch.randn(2, 1, 28, 28)
#     logits = model(x.view(2, 28, 28))
#     print(logits.shape)

import torch
import torch.nn as nn
from torch.optim import Adam

class BiLSTM(nn.Module):
    def __init__(self, d_emb, d_hid, vocab_size, n_layer, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_emb)
        self.enc = nn.LSTM(d_emb, d_hid, n_layer, dropout=dropout, bidirectional=True)
        self.dec = nn.Linear(d_hid*2, vocab_size)

    def forward(self, x, h0, c0):
        x = self.emb(x)
        seq_len, bsz, d_emb = x.shape
        # hs: (seq_len    , bsz, d_hid * 2)
        # h : (2 * n_layer, bsz, d_hid    )
        hs, (h,c) = self.enc(x, (h0, c0))
        logits = self.dec(hs)
        logits = logits.view(seq_len * bsz, -1)
        return logits, h


if __name__ == "__main__":
    # config
    d_emb, d_hid, vocab_size, n_layer = 8, 4, 16, 5
    bsz, lr = 2, 1e-4
    # data
    x = torch.Tensor([[1, 7],
                      [2, 8],
                      [3, 9]]).long()
    y = torch.Tensor([[2, 8],
                      [3, 9],
                      [4, 10]]).long()
    # model
    model = BiLSTM(d_emb, d_hid, vocab_size, n_layer)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    # training
    h0 = torch.zeros(n_layer*2, bsz, d_hid)
    c0 = torch.zeros(n_layer*2, bsz, d_hid)
    optimizer.zero_grad()
    logits, h = model(x, h0, c0)
    loss = loss_fn(logits, y.reshape(-1))
    loss.backward()
    optimizer.step()
    print(loss.item())