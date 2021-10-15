import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


def layer_init(layer):
  '''
  Initialize all weights and biases in layer and return it
  '''
  initrange = 0.1
  nn.init.uniform_(layer.weight.data, -initrange, initrange)  
  nn.init.constant_(layer.bias.data, 0)
  return layer


def repackage_hidden(hidden):
  '''
  Wraps hidden states in new Tensors, to detach them from their history.
  '''
  if isinstance(hidden, torch.Tensor):
    return hidden.detach()
  else:
    return tuple(repackage_hidden(v) for v in hidden)


def count_params(model):
  '''
  Count the number of parameters in model
  '''
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def embedded_dropout(embed, words, dropout=0.1):
  if dropout>0:
    mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
    masked_embed_weight = mask * embed.weight
  else:
    masked_embed_weight = embed.weight
  
  padding_idx = embed.padding_idx
  if padding_idx is None:
      padding_idx = 0

  X = F.embedding(
    words, masked_embed_weight, 
    padding_idx, embed.max_norm, embed.norm_type, 
    embed.scale_grad_by_freq, embed.sparse
  )
  return X

class LockedDropout(nn.Module):
  def __init__(self, batch_first=False):
    super().__init__()
    self.batch_first = batch_first

  def forward(self, x, dropout=0.5):
    '''
    Args: x (sequence length, batch size, rnn hidden size)
    '''
    if (dropout <= 0) or (not self.training):
      return x

    x = x.clone()
    if self.batch_first:
      mask = x.new_empty(x.size(0), 1, x.size(2), requires_grad=False).bernoulli_(1 - dropout)
    else:
      mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - dropout)
    mask = mask.div_(1 - dropout)
    mask = mask.expand_as(x)
    return mask * x


# copy from: https://github.com/salesforce/awd-lstm-lm/issues/86
class BackHook(torch.nn.Module):
  def __init__(self, hook):
    super(BackHook, self).__init__()
    self._hook = hook
    self.register_backward_hook(self._backward)

  def forward(self, *inp):
    return inp

  @staticmethod
  def _backward(self, grad_in, grad_out):
    self._hook()
    return None


class WeightDrop(torch.nn.Module):
  """
  Implements drop-connect, as per Merity et al https://arxiv.org/abs/1708.02182
  """
  def __init__(self, module, weights, dropout=0):
    super(WeightDrop, self).__init__()
    self.module = module
    self.weights = weights
    self.dropout = dropout
    self._setup()
    self.hooker = BackHook(lambda: self._backward())

  def _setup(self):
    for name_w in self.weights:
      w = getattr(self.module, name_w)
      self.register_parameter(name_w + '_raw', Parameter(w.data))

  def _setweights(self):
    for name_w in self.weights:
      raw_w = getattr(self, name_w + '_raw')
      if self.training:
        mask = raw_w.new_ones((raw_w.size(0), 1))
        mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
        w = mask.expand_as(raw_w) * raw_w
        setattr(self, name_w + "_mask", mask)
      else:
        w = raw_w
      rnn_w = getattr(self.module, name_w)
      rnn_w.data.copy_(w)

  def _backward(self):
    # transfer gradients from embeddedRNN to raw params
    for name_w in self.weights:
      raw_w = getattr(self, name_w + '_raw')
      rnn_w = getattr(self.module, name_w)
      raw_w.grad = rnn_w.grad * getattr(self, name_w + "_mask")

  def forward(self, *args):
    self._setweights()
    return self.module(*self.hooker(*args))