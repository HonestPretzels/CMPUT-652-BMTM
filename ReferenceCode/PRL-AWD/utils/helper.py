import os
import sys
import random
import datetime

import torch
import psutil
import numpy as np
from torch import nn
from sklearn import metrics
import torch.nn.functional as F
from collections import defaultdict

def Accuracy(state_value, obs):
  ''' Shape
  state_value   - (seq_len, batch_size, task_output_size)
  obs           - (seq_len, batch_size)
  '''
  y_pred = torch.argmax(state_value, dim=2)
  y_true = obs
  result = y_pred==y_true
  return result.float().mean().item()


def get_loss_func(task, GVF, GVF_discount, emb_size):
  ''' Shape
  state_value   - (seq_len, batch_size, task_output_size)
  obs           - (seq_len, batch_size)
  '''
  if GVF: # use TDError
    def TDError(state_value, obs):
      # We throw last element for one-step TD method
      task_output_size = state_value.size(2)
      # Get cumulant
      cumulants = F.one_hot(obs[:-1,:], num_classes=task_output_size).to(torch.bool)
      # Compute Q
      q = state_value[:-1,:,:]
      # Compute target Q
      q_target = torch.zeros_like(state_value[1:,:,:], device=state_value.device).to(torch.float32)
      values, _ = state_value[1:,:,:].clone().detach().max(2)
      values = values.view(-1)
      q_target[cumulants] = torch.ones_like(values) + GVF_discount * values
      q_target = q_target.clone().detach()
      # Get td_error
      td_error = nn.MSELoss(reduction='mean')(q, q_target)
      return td_error
    return TDError
  else: # use CrossEntropyLoss
    if task in ['Wiki2_POS', 'PTB_POS']:
      def CE(state_value, obs):
        state_value = state_value.view(-1, state_value.size(2))
        obs = obs.contiguous().view(-1)
        return nn.CrossEntropyLoss(reduction='mean')(state_value, obs)
      return CE
    elif task in ['PennTreebank', 'WikiText2', 'WikiText103']:
      return SplitCrossEntropyLoss(emb_size, [])


def get_perf_func(task, GVF, GVF_discount):
  ''' Shape
  state_value   - (seq_len, batch_size, task_output_size)
  obs           - (seq_len, batch_size)
  '''
  if task in ['PTB_POS', 'Wiki2_POS']:
    return Accuracy  
  elif task in ['PennTreebank', 'WikiText2', 'WikiText103']:
    def CE(state_value, obs):
      q_value = state_value.view(-1, state_value.size(2))
      obs = obs.contiguous().view(-1)
      return nn.CrossEntropyLoss(reduction='mean')(q_value, obs).item()
    return CE


def compute_v_backward(obs, gamma, label_size):
  ''' Shape
  v_backward    - (seq_len, batch_size, label_size)
  obs           - (seq_len, batch_size)
  '''
  seq_len = obs.size(0)
  cumulants = F.one_hot(obs, num_classes=label_size).to(torch.float)
  v_backward = cumulants.clone().detach()
  for i in range(1, seq_len):
    v_backward[i,:,:] = cumulants[i,:,:] + gamma * v_backward[i-1,:,:]
  return v_backward


def one_hot(idx, size, device):
  a = np.zeros((1, size), np.float32)
  a[0][idx] = 1
  v = to_tensor(a, device)
  return v


def get_time_str():
  return datetime.datetime.now().strftime("%y.%m.%d-%H:%M:%S")


def rss_memory_usage():
  '''
  Return the resident memory usage in MB
  '''
  process = psutil.Process(os.getpid())
  mem = process.memory_info().rss / float(2 ** 20)
  return mem


def str_to_class(module_name, class_name):
  '''
  Convert string to class
  '''
  return getattr(sys.modules[module_name], class_name)


def set_one_thread():
  '''
  Set number of threads for pytorch to 1
  '''
  os.environ['OMP_NUM_THREADS'] = '1'
  os.environ['MKL_NUM_THREADS'] = '1'
  torch.set_num_threads(1)


def to_tensor(x, device):
  '''
  Convert an array to tensor
  '''
  if isinstance(x, torch.Tensor):
    return x
  x = np.asarray(x, dtype=np.float)
  x = torch.tensor(x, device=device, dtype=torch.float32)
  return x


def to_numpy(t):
  '''
  Convert a tensor to numpy
  '''
  return t.cpu().detach().numpy()


def set_random_seed(seed):
  '''
  Set all random seeds
  '''
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  
  '''
  Deterministic mode can have a performance impact, depending on your model.
  This means that due to the deterministic nature of the model,
  the processing speed can be lower than when the model is non-deterministic.
  '''
  if torch.cuda.is_available(): 
    torch.cuda.manual_seed_all(seed)
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    """

def make_dir(dir):
  if not os.path.exists(dir): 
    os.makedirs(dir, exist_ok=True)


class SplitCrossEntropyLoss(nn.Module):
  # Split Cross Entropy Loss calculates an approximate softmax
  def __init__(self, hidden_size, splits):
    # We assume splits is [0, split1, split2, N] where N >= |V|
    # For example, a vocab of 1000 words may have splits [0] + [100, 500] + [inf]
    super(SplitCrossEntropyLoss, self).__init__()
    self.hidden_size = hidden_size
    self.splits = [0] + splits + [100 * 1000000]
    self.nsplits = len(self.splits) - 1
    self.stats = defaultdict(list)
    # Each of the splits that aren't in the head require a pretend token, we'll call them tombstones
    # The probability given to this tombstone is the probability of selecting an item from the represented split
    if self.nsplits > 1:
      self.tail_vectors = nn.Parameter(torch.zeros(self.nsplits - 1, hidden_size))
      self.tail_bias = nn.Parameter(torch.zeros(self.nsplits - 1))

  def logprob(self, weight, bias, hiddens, splits=None, softmaxed_head_res=None):
    # First we perform the first softmax on the head vocabulary and the tombstones
    if softmaxed_head_res is None:
      start, end = self.splits[0], self.splits[1]
      head_weight = None if end - start == 0 else weight[start:end]
      head_bias = None if end - start == 0 else bias[start:end]
      # We only add the tombstones if we have more than one split
      if self.nsplits > 1:
        head_weight = self.tail_vectors if head_weight is None else torch.cat([head_weight, self.tail_vectors])
        head_bias = self.tail_bias if head_bias is None else torch.cat([head_bias, self.tail_bias])

      # Perform the softmax calculation for the word vectors in the head for all splits
      # We need to guard against empty splits as torch.cat does not like random lists
      head_res = torch.nn.functional.linear(hiddens, head_weight, bias=head_bias)
      softmaxed_head_res = torch.nn.functional.log_softmax(head_res, dim=-1)

    if splits is None:
      splits = list(range(self.nsplits))

    results = []
    running_offset = 0
    for idx in splits:
      # For those targets in the head (idx == 0) we only need to return their loss
      if idx == 0:
        results.append(softmaxed_head_res[:, :-(self.nsplits - 1)])
      # If the target is in one of the splits, the probability is the p(tombstone) * p(word within tombstone)
      else:
        start, end = self.splits[idx], self.splits[idx + 1]
        tail_weight = weight[start:end]
        tail_bias = bias[start:end]
        # Calculate the softmax for the words in the tombstone
        tail_res = torch.nn.functional.linear(hiddens, tail_weight, bias=tail_bias)
        # Then we calculate p(tombstone) * p(word in tombstone)
        # Adding is equivalent to multiplication in log space
        head_entropy = (softmaxed_head_res[:, -idx]).contiguous()
        tail_entropy = torch.nn.functional.log_softmax(tail_res, dim=-1)
        results.append(head_entropy.view(-1, 1) + tail_entropy)
    if len(results) > 1:
      return torch.cat(results, dim=1)
    return results[0]

  def split_on_targets(self, hiddens, targets):
    # Split the targets into those in the head and in the tail
    split_targets = []
    split_hiddens = []
    # Determine to which split each element belongs (for each start split value, add 1 if equal or greater)
    mask = None
    for idx in range(1, self.nsplits):
      partial_mask = targets >= self.splits[idx]
      mask = mask + partial_mask if mask is not None else partial_mask
    
    for idx in range(self.nsplits):
      # If there are no splits, avoid costly masked select
      if self.nsplits == 1:
        split_targets, split_hiddens = [targets], [hiddens]
        continue
      # If all the words are covered by earlier targets, we have empties so later stages don't freak out
      if sum(len(t) for t in split_targets) == len(targets):
        split_targets.append([])
        split_hiddens.append([])
        continue
      # Are you in our split?
      tmp_mask = mask == idx
      split_targets.append(torch.masked_select(targets, tmp_mask))
      split_hiddens.append(hiddens.masked_select(tmp_mask.unsqueeze(1).expand_as(hiddens)).view(-1, hiddens.size(1)))
    return split_targets, split_hiddens

  def forward(self, weight, bias, hiddens, targets):
    total_loss = None
    if len(hiddens.size()) > 2: hiddens = hiddens.view(-1, hiddens.size(2))

    split_targets, split_hiddens = self.split_on_targets(hiddens, targets)

    # First we perform the first softmax on the head vocabulary and the tombstones
    start, end = self.splits[0], self.splits[1]
    head_weight = None if end - start == 0 else weight[start:end]
    head_bias = None if end - start == 0 else bias[start:end]
    # We only add the tombstones if we have more than one split
    if self.nsplits > 1:
      head_weight = self.tail_vectors if head_weight is None else torch.cat([head_weight, self.tail_vectors])
      head_bias = self.tail_bias if head_bias is None else torch.cat([head_bias, self.tail_bias])
    # Perform the softmax calculation for the word vectors in the head for all splits
    # We need to guard against empty splits as torch.cat does not like random lists
    combo = torch.cat([split_hiddens[i] for i in range(self.nsplits) if len(split_hiddens[i])])
    all_head_res = torch.nn.functional.linear(combo, head_weight, bias=head_bias)
    softmaxed_all_head_res = torch.nn.functional.log_softmax(all_head_res, dim=-1)
    
    running_offset = 0
    for idx in range(self.nsplits):
      # If there are no targets for this split, continue
      if len(split_targets[idx]) == 0: continue
      # For those targets in the head (idx == 0) we only need to return their loss
      if idx == 0:
        softmaxed_head_res = softmaxed_all_head_res[running_offset:running_offset + len(split_hiddens[idx])]
        entropy = -torch.gather(softmaxed_head_res, dim=1, index=split_targets[idx].view(-1, 1))
      # If the target is in one of the splits, the probability is the p(tombstone) * p(word within tombstone)
      else:
        softmaxed_head_res = softmaxed_all_head_res[running_offset:running_offset + len(split_hiddens[idx])]
        # Calculate the softmax for the words in the tombstone
        tail_res = self.logprob(weight, bias, split_hiddens[idx], splits=[idx], softmaxed_head_res=softmaxed_head_res)
        # Then we calculate p(tombstone) * p(word in tombstone)
        # Adding is equivalent to multiplication in log space
        head_entropy = softmaxed_head_res[:, -idx]
        # All indices are shifted - if the first split handles [0,...,499] then the 500th in the second split will be 0 indexed
        indices = (split_targets[idx] - self.splits[idx]).view(-1, 1)
        # Warning: if you don't squeeze, you get an N x 1 return, which acts oddly with broadcasting
        tail_entropy = torch.gather(torch.nn.functional.log_softmax(tail_res, dim=-1), dim=1, index=indices).squeeze()
        entropy = -(head_entropy + tail_entropy)

      running_offset += len(split_hiddens[idx])
      total_loss = entropy.float().sum() if total_loss is None else total_loss + entropy.float().sum()

    return (total_loss / len(targets)).type_as(weight)