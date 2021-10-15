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



# For GVFN, we use action value Q(s,a) for implementation in this file.


def Accuracy(state_value, obs):
  ''' Shape
  state_value   - (seq_len, batch_size, task_output_size)
  obs           - (seq_len, batch_size)
  '''
  y_pred = torch.argmax(state_value, dim=2)
  y_true = obs
  result = y_pred==y_true
  return result.float().mean().item()


def F1_score(state_value, obs):
  ''' Shape
  state_value   - (seq_len, batch_size, task_output_size)
  obs           - (seq_len, batch_size)
  '''
  y_pred = torch.argmax(state_value, dim=2)
  y_true = obs
  y_pred, y_true = to_numpy(y_pred).flatten(), to_numpy(y_true).flatten()
  F1 = metrics.f1_score(y_true, y_pred, average='macro')
  if np.isnan(F1): F1 = 0.0
  '''
  print(f'Precision: {metrics.precision_score(y_true, y_pred, average="macro"):.4f}')
  print(f'Recall: {metrics.recall_score(y_true, y_pred, average="macro"):.4f}')
  '''
  return F1


def get_loss_func(task, GVF, GVF_discount):
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
    if task in ['Wiki2_POS', 'PTB_POS', 'UDPOS', 'CoNLL2000Chunking', 'CoNLL2003_NER', 'CoNLL2003_POS', 'CoNLL2003_CHUNK']:
      def CE(state_value, obs):
        state_value = state_value.view(-1, state_value.size(2))
        obs = obs.contiguous().view(-1)
        return nn.CrossEntropyLoss(reduction='mean')(state_value, obs)
      return CE
    elif task in ['PennTreebank', 'WikiText2', 'WikiText103']:
      def NLL(log_prob, obs):
        log_prob = log_prob.view(-1, log_prob.size(2))
        obs = obs.contiguous().view(-1)
        return F.nll_loss(log_prob, obs)
      return NLL


def get_perf_func(task, GVF, GVF_discount):
  ''' Shape
  state_value   - (seq_len, batch_size, task_output_size)
  obs           - (seq_len, batch_size)
  '''
  if task in ['UDPOS', 'PTB_POS', 'Wiki2_POS']:
    return Accuracy  
  elif task in ['CoNLL2000Chunking', 'CoNLL2003', 'CoNLL2003_NER', 'CoNLL2003_POS', 'CoNLL2003_CHUNK']:
    return F1_score
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