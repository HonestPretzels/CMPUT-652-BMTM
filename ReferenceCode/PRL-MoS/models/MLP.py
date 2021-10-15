import torch
import torch.nn as nn
import torch.nn.functional as F
from models.helper import layer_init


class MLP(nn.Module):
  '''
  Multilayer Perceptron
  '''
  def __init__(self, layer_dims, hidden_activation=nn.ReLU(), output_activation=None):
    super().__init__()
    # Create layers
    self.mlp = nn.ModuleList([])
    for i in range(len(layer_dims[:-1])):
      dim_in, dim_out = layer_dims[i], layer_dims[i+1]
      self.mlp.append(layer_init(nn.Linear(dim_in, dim_out, bias=True)))
      if i+2 != len(layer_dims):
        if hidden_activation is not None:
          self.mlp.append(hidden_activation)
      elif output_activation is not None:
        self.mlp.append(output_activation)

  
  def forward(self, x):
    for layer in self.mlp:
      x = layer(x)
    return x