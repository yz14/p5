---
layout: post
title:  "Autoencoder"
---

Autoencoder 是一种无监督学习模型，用于数据压缩和特征学习。它由一个编码器（encoder）和一个解码器（decoder）组成，通常是一个多层神经网络。编码器将输入数据压缩成低维表示（通常称为编码或隐藏表示），而解码器则将该低维表示映射回原始输入数据空间。  

{% highlight ruby %}
import torch
import torch.nn as nn
class Autoencoder(torch.nn.Module):

    def __init__(self, num_features):
        super(Autoencoder, self).__init__()
        
        ### ENCODER
        self.linear_1 = torch.nn.Linear(num_features, num_hidden_1)
        # The following to lones are not necessary, 
        # but used here to demonstrate how to access the weights
        # and use a different weight initialization.
        # By default, PyTorch uses Xavier/Glorot initialization, which
        # should usually be preferred.
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()
        
        ### DECODER
        self.linear_2 = torch.nn.Linear(num_hidden_1, num_features)
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()
        

    def forward(self, x):
        
        ### ENCODER
        encoded = self.linear_1(x)
        encoded = F.leaky_relu(encoded)
        
        ### DECODER
        logits = self.linear_2(encoded)
        decoded = torch.sigmoid(logits)
        
        return decoded
{% endhighlight %}  
