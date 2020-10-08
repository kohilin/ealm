import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
import torch
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def key_value_attention(key, value):
    product = key.matmul(value.transpose(0, 1))
    relued_product = torch.relu(product)
    attn = relued_product / relued_product.sum(dim=1, keepdim=True)
    return attn.matmul(value)


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, layer_num=2, activation=nn.ReLU):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.lay_num = layer_num
        self.activation = activation
        layers = [nn.Linear(in_dim, hid_dim)]
        for i in range(self.lay_num-1):
            layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            layers.append(self.activation())
        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.Sequential(*layers)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, x, stop_layer=None):
        for l in self.layers[:stop_layer]:
            x = l(x)
        return x
