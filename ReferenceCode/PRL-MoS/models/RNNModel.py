import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.MLP import MLP
from models.helper import LockedDropout, embedded_dropout, WeightDrop
from models.helper import layer_init, repackage_hidden


activations = {
  'None': None,
  'ReLU': nn.ReLU(),
  'LeakyReLU': nn.LeakyReLU(),
  'Tanh': nn.Tanh(),
  'Sigmoid': nn.Sigmoid()
}


class Encoder(nn.Module):
  '''
  The shared encoder.
  '''
  def __init__(self, cfg, vocab_size):
    super(Encoder, self).__init__()
    self.use_dropout = True
    self.emb_size = cfg['embedding']['size']
    # Dropout
    self.dropout_emb = cfg['dropout_emb']
    self.lockdrop = LockedDropout()
    # Define embedding encoder
    self.encoder = nn.Embedding(vocab_size, self.emb_size)
    self.encoder.weight.data.uniform_(-0.1, 0.1)

  def forward(self, x):
    batch_size = x.size(1)
    # Get word embedding
    x_emb = embedded_dropout(self.encoder, x, dropout=self.dropout_emb if (self.training and self.use_dropout) else 0)
    return x_emb


class LMDecoder(nn.Module):
  """
  The decoder for language modeling.
  """
  def __init__(self, cfg, encoder, task_output_sizes, aux_layer_select):
    super(LMDecoder, self).__init__()
    self.use_dropout = True
    self.tasks = cfg['tasks']
    self.aux_layer_select = aux_layer_select
    # RNN
    self.rnn_type = cfg['rnn']['type']
    self.num_layers = cfg['rnn']['num_layers']
    self.num_experts = cfg['rnn']['num_experts']
    # Sizes
    self.emb_size = cfg['embedding']['size']
    rnn_input_size = self.emb_size
    self.hidden_last = cfg['rnn']['hidden_last']
    self.vocab_size = task_output_sizes['word']
    self.aux_input_size = task_output_sizes['label']
    self.old_hidden_sizes = [cfg['rnn']['hidden_size']] * (self.num_layers - 1) + [cfg['rnn']['hidden_last']]
    self.new_hidden_sizes = [cfg['rnn']['hidden_size']] * (self.num_layers - 1) + [cfg['rnn']['hidden_last']]
    if len(self.tasks) == 2:
      self.new_hidden_sizes[aux_layer_select] += task_output_sizes['label']
    self.hidden_last = self.new_hidden_sizes[-1]
    # Dropouts
    self.dropout_in  = cfg['dropout_in']
    self.dropout_we  = cfg['dropout_we']
    self.dropout_hd  = cfg['dropout_hd']
    self.dropout_out = cfg['dropout_out']
    self.dropout_lt = cfg['dropout_lt']
    self.lockdrop = LockedDropout()
    # Define RNNs
    assert self.rnn_type in ['LSTM', 'GRU'], 'RNN type is not supported'
    self.rnns = [
      getattr(nn, self.rnn_type)(
        input_size=rnn_input_size if l==0 else self.new_hidden_sizes[l-1],
        hidden_size=self.old_hidden_sizes[l],
        num_layers=1,
        dropout=0,
        batch_first=False
      )
      for l in range(self.num_layers)
    ]
    if self.dropout_we > 0:
      self.rnns = [
        WeightDrop(rnn, ['weight_hh_l0'], dropout=self.dropout_we if self.use_dropout else 0)
        for rnn in self.rnns
      ]
    self.rnns = nn.ModuleList(self.rnns)
    
    # Define LM decoder
    self.prior = nn.Linear(self.hidden_last, self.num_experts, bias=False)
    self.latent = nn.Sequential(nn.Linear(self.hidden_last, self.num_experts*self.emb_size), nn.Tanh())
    self.decoder = layer_init(nn.Linear(self.emb_size, self.vocab_size))
    if cfg['embedding']['shared_embedding']:
      self.decoder.weight = encoder.weight


  def forward(self, x_emb, hidden, auxiliary_input):
    ''' Shape:
    x             - (seq_len, batch_size)
    hidden        - (num_layers, num_directions, batch_size, rnn_hidden_size)
    x_emb         - (seq_len, batch_size, embed_size)
    output        - (seq_len, batch_size, num_directions * rnn_hidden_size)
    new_h         - (num_directions, batch_size, rnn_hidden_size)
    new_hidden    - (num_layers, num_directions, batch_size, rnn_hidden_size)
    state_value   - (seq_len, batch_size, task_output_size)
    '''    
    for rnn in self.rnns:
      rnn.module.flatten_parameters()
    
    x_emb = self.lockdrop(x_emb, self.dropout_in if self.use_dropout else 0)
    batch_size = x_emb.size(1)
    hidden = repackage_hidden(hidden)
    # Encode embedding by RNN
    raw_output = x_emb
    new_hidden = []
    outputs, raw_outputs = [], []
    for l, rnn in enumerate(self.rnns):
      raw_output, new_h = rnn(raw_output, hidden[l])
      new_hidden.append(new_h)
      raw_outputs.append(raw_output)
      if l != self.num_layers - 1:
        raw_output = self.lockdrop(raw_output, self.dropout_hd if self.use_dropout else 0)
        outputs.append(raw_output)
        # Generate new raw_output with concatenations
        if len(self.tasks) == 2 and self.aux_layer_select == l:
          auxiliary_input = auxiliary_input.clone().detach()
          raw_output = torch.cat([raw_output, auxiliary_input], 2)

    # Generate input for LM decoder
    output = self.lockdrop(raw_output, self.dropout_out if self.use_dropout else 0)
    outputs.append(output)
    if len(self.tasks) == 2 and self.aux_layer_select == self.num_layers - 1:
      auxiliary_input = auxiliary_input.clone().detach()
      output = torch.cat([output, auxiliary_input], 2)
    # Get latent variables
    latent = self.latent(output)
    latent = self.lockdrop(latent, self.dropout_lt if self.use_dropout else 0)
    logit = self.decoder(latent.view(-1, self.emb_size))
    # Get probability weight
    prior_logit = self.prior(output).contiguous().view(-1, self.num_experts)
    prior = F.softmax(prior_logit, -1)
    # Get weighted probability
    prob = F.softmax(logit.view(-1, self.vocab_size), -1).view(-1, self.num_experts, self.vocab_size)
    prob = (prob * prior.unsqueeze(2).expand_as(prob)).sum(1)
    log_prob = torch.log(prob.add_(1e-8)).view(-1, batch_size, self.vocab_size) # log_prob

    return log_prob, new_hidden, raw_outputs, outputs


  def init_hidden(self, batch_size):
    hidden_list = []
    weight = next(self.parameters()).data
    if self.rnn_type == 'LSTM':
      for l in range(self.num_layers):
        hidden_list.append(
          (weight.new_zeros(1, batch_size, self.old_hidden_sizes[l]),
           weight.new_zeros(1, batch_size, self.old_hidden_sizes[l]))
        )
    else:
      for l in range(self.num_layers):
        hidden_list.append(
          weight.new_zeros(1, batch_size, self.old_hidden_sizes[l])
        )
    return hidden_list


class AuxDecoder(nn.Module):
  '''
  The decoder for auxiliary tasks.
  '''
  def __init__(self, cfg, label_size):
    super(AuxDecoder, self).__init__()
    self.use_dropout = True
    # RNN
    self.rnn_type = cfg['rnn']['type']
    self.num_layers = cfg['rnn']['num_layers']
    # Sizes
    self.gamma = cfg['gamma']
    if self.gamma < 0:
      rnn_input_size = cfg['embedding']['size']
    else:
      rnn_input_size = cfg['embedding']['size'] + label_size
    self.hidden_size = cfg['rnn']['hidden_size']
    # Dropouts
    self.dropout_in  = cfg['dropout_in']
    self.dropout_we  = cfg['dropout_we']
    self.dropout_hd  = cfg['dropout_hd']
    self.dropout_out = cfg['dropout_out']
    self.lockdrop = LockedDropout()
    # Define RNNs
    assert self.rnn_type in ['LSTM', 'GRU'], 'RNN type is not supported'
    self.rnns = [
      getattr(nn, self.rnn_type)(
        input_size=rnn_input_size if l==0 else self.hidden_size,
        hidden_size=self.hidden_size,
        num_layers=1,
        dropout=0,
        batch_first=False
      )
      for l in range(self.num_layers)
    ]
    if self.dropout_we > 0:
      self.rnns = [
        WeightDrop(rnn, ['weight_hh_l0'], dropout=self.dropout_we if self.use_dropout else 0)
        for rnn in self.rnns
      ]
    self.rnns = nn.ModuleList(self.rnns)
    # Define auxiliary decoder
    MLP_layers = [self.hidden_size] + cfg['mlp'] + [label_size]
    self.decoder = MLP(layer_dims=MLP_layers, hidden_activation=activations[cfg['hidden_activation']], output_activation=activations[cfg['output_activation']])


  def forward(self, x_emb, hidden, v_backward):
    for rnn in self.rnns:
      rnn.module.flatten_parameters()

    x_emb = self.lockdrop(x_emb, self.dropout_in if self.use_dropout else 0)
    batch_size = x_emb.size(1)
    hidden = repackage_hidden(hidden)
    # Encode embedding by RNN
    if self.gamma < 0:
      raw_output = x_emb
    else:  
      raw_output = torch.cat([x_emb, v_backward], 2)
    for l, rnn in enumerate(self.rnns):
      raw_output, _ = rnn(raw_output, hidden[l])
      if l != self.num_layers - 1:
        raw_output = self.lockdrop(raw_output, self.dropout_hd if self.use_dropout else 0)

    state = self.lockdrop(raw_output, self.dropout_out if self.use_dropout else 0)
    state_value = self.decoder(state)
    return state_value


  def init_hidden(self, batch_size):
    hidden_list = []
    weight = next(self.parameters()).data
    if self.rnn_type == 'LSTM':
      for l in range(self.num_layers):
        hidden_list.append(
          (weight.new_zeros(1, batch_size, self.hidden_size),
           weight.new_zeros(1, batch_size, self.hidden_size))
        )
    else:
      for l in range(self.num_layers):
        hidden_list.append(
          weight.new_zeros(1, batch_size, self.hidden_size)
        )
    return hidden_list