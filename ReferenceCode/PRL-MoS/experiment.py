import os
import gc
import sys
import copy
import time
import json
import math
import torch
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F

from utils.helper import *
from utils.logger import Logger
from utils.dataloader import load_data, get_batch

from models.MLP import MLP
from models.helper import count_params
from models.RNNModel import Encoder, LMDecoder, AuxDecoder


aux_cfg = {
  'PennTreebank': {
      "dropout_in": 0.725,
      "dropout_we": 0.225,
      "dropout_hd": 0.225,
      "dropout_out": 0.4,
      "rnn": {
        "type": "LSTM",
        "hidden_size": 380,
        "num_layers": 2
      }
  },
  'WikiText2': {
      "dropout_in": 0.725,
      "dropout_we": 0.225,
      "dropout_hd": 0.225,
      "dropout_out": 0.4,
      "rnn": {
        "type": "LSTM",
        "hidden_size": 380,
        "num_layers": 2
      }
  }
}


class Experiment(object):
  def __init__(self, cfg):
    self.cfg = copy.deepcopy(cfg)
    if torch.cuda.is_available() and 'cuda' in cfg['device']:
      self.device = cfg['device']
    else:
      self.cfg['device'] = 'cpu'
      self.device = 'cpu'
    if self.cfg['generate_random_seed'] and self.cfg['mode'] != 'Dynamic':
      self.cfg['seed'] = np.random.randint(int(1e6))
    
    self.results = {}
    for task in cfg['tasks']:
      self.results[task] = {}
      for mode in ['Train', 'Test', 'Valid', 'Dynamic']:
        self.results[task][mode] = []
    
    self.log_paths = lambda task, mode: cfg['logs_dir'] + f'result_{task}_{mode}.feather'
    self.model_path = lambda task: cfg['logs_dir'] + f'model_{task}.pt'
    
    if self.cfg['mode'] == 'Dynamic':
      self.logger = Logger(cfg['logs_dir'], file_name='log_dyna.txt', filemode='a')
    elif self.cfg['mode'] == 'Test':
      self.logger = Logger(cfg['logs_dir'], file_name='log_test.txt')
    else:
      self.logger = Logger(cfg['logs_dir'])

    self.tasks = cfg['tasks']
    self.main_task = self.tasks[-1]
    if len(self.tasks) == 2:
      self.aux_task = self.tasks[0]
      self.layer_select = cfg['layer_select'] - 1
    else:
      self.aux_task = None
      self.layer_select = -100
    assert self.main_task in ['PennTreebank', 'WikiText2', 'WikiText103'], 'Main task must be language modeling!'
        
    self.config_idx = cfg['config_idx']
    self.GVF_discounts = cfg['GVF_discounts']
    self.gamma = cfg['gamma']
    self.nonmono = cfg['optimizer']['nonmono']
    
    assert self.cfg['mode'] == 'Dynamic' or cfg['batch_size'] % cfg['small_batch_size'] == 0, 'batch_size must be divisible by small_batch_size'
    self.batch_sizes = {
      'Train': cfg['batch_size'],
      'Valid': 10,
      'Test': 1,
      'Small': cfg['small_batch_size']
    }
    self.train_epochs = cfg['train_epochs']
    self.mode = cfg['mode']
    self.bptt_len = cfg['bptt_len']
    self.bptt_len_delta = cfg['bptt_len_delta']
    if self.mode == 'Train':
      self.save_config()
    
    self.GVFs = {}
    self.rnn_type = {}
    self.best_epoch_idx = {}
    self.best_valid_perf = {}
    for task in self.tasks:
      self.best_epoch_idx[task] = -1
      self.best_valid_perf[task] = 0
      if self.GVF_discounts[task] < 0:
        self.GVFs[task] = False
        self.rnn_type[task] = cfg['rnn']['type']
      else:
        self.GVFs[task] = True
        self.rnn_type[task] = 'GVFN_' + cfg['rnn']['type']


  def run(self):
    set_one_thread()
    set_random_seed(self.cfg['seed'])
    self.start_time = time.time()
    self.logger.info('Load datasets ...')
    if 'PennTreebank' in self.tasks:
      dataset = 'PTB_POS'
    elif 'WikiText2' in self.tasks:
      dataset = 'Wiki2_POS'
    else:
      dataset = None
    self.data, self.task_output_sizes = load_data(
      dataset=dataset,
      batch_sizes=self.batch_sizes,
      device=self.device
    )
    self.logger.info(f'Tasks: {self.tasks}')
    self.logger.info(f'Task output sizes: {self.task_output_sizes}')
    mode = 'Train'
    self.logger.info(f'Number of batches for {self.main_task} {mode}: {len(self.data[mode]["text"])//self.bptt_len}')
    
    self.logger.info('Build Model: Encoder ...')
    self.encoder = Encoder(self.cfg, self.task_output_sizes['word']).to(self.device)
    model_size = 0
    
    self.logger.info('Build loss funcs, performance funcs, decoders, and optimizers ...')
    self.params = {}
    self.decoders = {}
    self.loss_funcs = {}
    self.perf_funcs = {}
    self.optimizers = {}
    for task in self.tasks:
      self.loss_funcs[task] = get_loss_func(task, self.GVFs[task], self.GVF_discounts[task])
      self.perf_funcs[task] = get_perf_func(task, self.GVFs[task], self.GVF_discounts[task])
      if task == self.main_task:
        self.logger.info('Build Model: LM Decoder ...')
        self.decoders[task] = LMDecoder(self.cfg, self.encoder.encoder, self.task_output_sizes, self.layer_select).to(self.device)
        if self.mode == 'Dynamic':
          self.params[task] = list(self.decoders[task].parameters())
        else:
          self.params[task] = list(self.encoder.parameters()) + list(self.decoders[task].parameters())
      else:
        aux_cfg[self.main_task]['gamma'] = self.cfg['gamma']
        aux_cfg[self.main_task]['mlp'] = self.cfg['mlp']
        aux_cfg[self.main_task]['embedding'] = self.cfg['embedding']
        aux_cfg[self.main_task]['hidden_activation'] = self.cfg['hidden_activation']
        aux_cfg[self.main_task]['output_activation'] = self.cfg['output_activation']
        self.logger.info('Build Model: Auxiliary Decoder ...')
        self.decoders[task] = AuxDecoder(aux_cfg[self.main_task], self.task_output_sizes['label']).to(self.device)
        self.params[task] = list(self.encoder.parameters()) + list(self.decoders[task].parameters())
      self.optimizers[task] = getattr(torch.optim, self.cfg['optimizer']['name'])(self.params[task], lr=self.cfg['optimizer']['lr'][task], weight_decay=self.cfg['optimizer']['weight_decay'])
      model_size += count_params(self.decoders[task])/1e6
    self.logger.info(f'Model size: {model_size:.2f} M')

    self.logger.info('Start running ...')
    if self.mode == 'Train': # Train && Valid
      self.train()
    if self.mode in ['Train', 'Test']: # Test
      for task in self.tasks:
        _ = self.evaluate('Test', task, self.best_epoch_idx[task])
    if self.mode == 'Dynamic':
      self.dynamic_eval()
    # Prepare for end
    end_time = time.time()
    self.logger.info(f'Memory usage: {rss_memory_usage():.2f} MB')
    self.logger.info(f'Time elapsed: {(end_time-self.start_time)/60:.2f} minutes')


  def train(self, mode='Train'):
    '''
    Train for multiple epochs and valid
    '''
    best_val_perf = []
    total_seq_len = len(self.data[mode]['text'])
    # Initailization hidden states
    self.hiddens = {}
    for task in self.tasks:
      if task == self.main_task:
        self.hiddens[task] = [self.decoders[task].init_hidden(self.batch_sizes['Small']) for _ in range(self.batch_sizes[mode] // self.batch_sizes['Small'])]
      else:
        self.hiddens[task] = self.decoders[task].init_hidden(self.batch_sizes[mode])
        self.hiddens['aux_for_main'] = [self.decoders[task].init_hidden(self.batch_sizes['Small']) for _ in range(self.batch_sizes[mode] // self.batch_sizes['Small'])]
    # Start training
    for i in range(1, self.train_epochs+1):
      self.set_model_mode(mode)
      location = 0
      epoch_loss, epoch_perf = {}, {}
      for task in self.tasks:
        epoch_loss[task], epoch_perf[task] = 0.0, 0.0
      while location < self.data[mode]['text'].size(0) - 1 - 1:
        # Get sequence length
        bptt_len = self.bptt_len if np.random.random() < 0.95 else self.bptt_len / 2.0
        seq_len = max(5, int(np.random.normal(bptt_len, 5))) # Prevent a very small sequence length
        seq_len = min(seq_len, self.bptt_len + self.bptt_len_delta) # Prevent a very long sequence length
        # Change lr for main task
        self.optimizers[self.main_task].param_groups[0]['lr'] = self.cfg['optimizer']['lr'][self.main_task] * seq_len / self.bptt_len
        # get batch
        text, next_text, label, next_label = get_batch(self.data[mode], location, self.bptt_len, seq_len=seq_len)
        # Change location
        location += seq_len
        # Prevent excessively small sequence length or wrong batch size
        if text.size(0) < 3 or text.size(1) != self.batch_sizes[mode]:
          continue
        # Run for one batch
        loss, perf = self.train_one_batch(text, next_text, label, next_label)
        for task in self.tasks:
          epoch_loss[task] += loss[task] * len(text)
          epoch_perf[task] += perf[task] * len(text)
      
      # Display train result && save result
      for task in self.tasks:
        epoch_loss[task] /= total_seq_len
        epoch_perf[task] /= total_seq_len
        result_dict = {
          'Epoch': i,
          'RNN': self.rnn_type[task],
          'Task': task,
          'Loss': epoch_loss[task],
          'Perf': epoch_perf[task]
        }
        self.results[task][mode].append(result_dict)
        self.save_results(task, mode)
        self.logger.info(f'<{self.config_idx}> {task} ({self.rnn_type[task]}) [{mode}] Epoch {i}/{self.train_epochs}, Loss={epoch_loss[task]:.4f}, Perf={epoch_perf[task]:.4f}')
      
      # Validation
      if 't0' in self.optimizers[self.main_task].param_groups[0]:
        tmp = {}
        for prm in self.params[self.main_task]:
          tmp[prm] = prm.data.clone()
          prm.data = self.optimizers[self.main_task].state[prm]['ax'].clone()
        _ = self.evaluate('Valid', self.main_task, i)
        for prm in self.params[self.main_task]:
          prm.data = tmp[prm].clone()
      else:
        val_perf = self.evaluate('Valid', self.main_task, i)
        if ('t0' not in self.optimizers[self.main_task].param_groups[0]) and \
        (len(best_val_perf) > self.nonmono) and (val_perf < max(best_val_perf[:-self.nonmono])):
          self.logger.info('Switching to ASGD')
          self.optimizers[self.main_task] = torch.optim.ASGD(self.params[self.main_task], lr=self.cfg['optimizer']['lr'][self.main_task], t0=0, lambd=0, weight_decay=self.cfg['optimizer']['weight_decay'])
        best_val_perf.append(val_perf)
      if self.aux_task is not None:
        _ = self.evaluate('Valid', self.aux_task, i)

      # Display speed
      self.logger.info(f"<{self.config_idx}> Speed={(time.time()-self.start_time)/i:.2f} (s/epoch)")
        

  def train_one_batch(self, text, next_text, label, next_label):
    mode = 'Train'
    loss, perf = {}, {}
    for task in self.tasks:
      loss[task], perf[task] = 0.0, 0.0
      self.optimizers[task].zero_grad()

    # ------------> main task <------------
    task = self.main_task
    hidden = self.hiddens[task]
    new_hidden = []
    start, end, s_id = 0, self.batch_sizes['Small'], 0
    while start < self.batch_sizes[mode]:
      x, next_x = text[:, start: end], next_text[:, start: end]
      y, next_y = label[:, start: end], next_label[:, start: end]
      v_backward = compute_v_backward(y, self.gamma, self.task_output_sizes['label'])
      # Foward
      x_emb = self.encoder(x)
      if self.aux_task is not None:
        auxiliary_input = self.decoders[self.aux_task](x_emb, self.hiddens['aux_for_main'][s_id], v_backward)
        if self.GVF_discounts[self.aux_task] < 0:
          auxiliary_input = F.softmax(auxiliary_input, -1)
      else:
        auxiliary_input = None
      log_prob, new_h, rnn_hs, dropped_rnn_hs = self.decoders[task](x_emb, hidden[s_id], auxiliary_input)
      new_hidden.append(new_h)
      # Get loss
      loss_tensor = self.loss_funcs[task](log_prob, next_x)
      loss[task] += loss_tensor.item() * self.batch_sizes['Small'] / self.batch_sizes[mode]
      # Activiation Regularization
      if self.cfg['optimizer']['alpha']>0:
        loss_tensor += sum(self.cfg['optimizer']['alpha'] * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
      # Temporal Activation Regularization
      if self.cfg['optimizer']['beta']>0:
        loss_tensor += sum(self.cfg['optimizer']['beta'] * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
      # Backward
      loss_tensor *= self.batch_sizes['Small'] / self.batch_sizes[mode]
      loss_tensor.backward()
      # Get performance measurement
      if self.GVFs[task]:
        perf[task] += -1.0*self.perf_funcs[task](log_prob, next_x) * self.batch_sizes['Small'] / self.batch_sizes[mode]
      else:
        perf[task] = -1.0 * loss[task]
      # Update index
      s_id += 1
      start = end
      end = start + self.batch_sizes['Small']
    # Take a optimization step
    if self.cfg['optimizer']['gradient_clip']>0: # Clip the gradients
      torch.nn.utils.clip_grad_norm_(self.params[task], self.cfg['optimizer']['gradient_clip'])
    self.optimizers[task].step()
    # Update hidden states
    self.hiddens[task] = new_hidden

    # ------------> auxiliary task <------------
    if self.aux_task is not None:
      task = self.aux_task
      # Foward
      x_emb = self.encoder(text)
      v_backward = compute_v_backward(label, self.gamma, self.task_output_sizes['label'])
      state_value = self.decoders[task](x_emb, self.hiddens[task], v_backward)
      # Get loss
      loss_tensor = self.loss_funcs[task](state_value, next_label)
      loss[task] = loss_tensor.item()
      # Backward
      loss_tensor.backward()
      # Get performance measurement
      perf[task] = self.perf_funcs[task](state_value, next_label)
      # Take a optimization step
      if self.cfg['optimizer']['gradient_clip']>0: # Clip the gradients
        torch.nn.utils.clip_grad_norm_(self.params[task], self.cfg['optimizer']['gradient_clip'])
      self.optimizers[task].step()

    return loss, perf


  def evaluate(self, mode, task, epoch_idx=-1):
    assert mode in ['Test', 'Valid']
    epoch_loss, epoch_perf = 0.0, 0.0
    # Load model for test
    if mode == 'Test': self.load_model(task)
    # Set model to evaluation mode
    self.set_model_mode(mode)
    # Initailization evaluation hidden states
    eval_hidden = {}
    for t in self.tasks:
      eval_hidden[t] = self.decoders[t].init_hidden(self.batch_sizes[mode])
    total_seq_len = len(self.data[mode]['text'])
    with torch.no_grad():
      for location in range(0, self.data[mode]['text'].size(0) - 1, self.bptt_len):
        text, next_text, label, next_label = get_batch(self.data[mode], location, self.bptt_len)
        # Prevent excessively small sequence length or wrong batch size
        if text.size(0) < 3 or text.size(1) != self.batch_sizes[mode]:
          continue
        # Foward
        x_emb = self.encoder(text)
        v_backward = compute_v_backward(label, self.gamma, self.task_output_sizes['label'])
        if self.aux_task is not None:
          state_value = self.decoders[self.aux_task](x_emb, eval_hidden[self.aux_task], v_backward)
        # Get loss && performance measurement
        if task == self.main_task:
          if self.aux_task is not None:
            if self.GVF_discounts[self.aux_task] < 0:
              auxiliary_input = F.softmax(state_value, -1)
            else:
              auxiliary_input = state_value
          else:
            auxiliary_input = None
          # Forward
          log_prob, eval_hidden[task], _, _ = self.decoders[task](x_emb, eval_hidden[task], auxiliary_input)
          # Get loss
          loss = self.loss_funcs[task](log_prob, next_text).item()
          if self.GVFs[task]:
            perf = -1.0*self.perf_funcs[task](log_prob, next_text)
          else:
            perf = -1.0*loss
        else:
          loss = self.loss_funcs[task](state_value, next_label).item()
          perf = self.perf_funcs[task](state_value, next_label)
        # Add epoch loss && perf
        epoch_loss += loss * len(text)
        epoch_perf += perf * len(text)
    
    # Save result
    epoch_loss /= total_seq_len
    epoch_perf /= total_seq_len
    result_dict = {
      'Epoch': epoch_idx,
      'RNN': self.rnn_type[task],
      'Task': task,
      'Loss': epoch_loss,
      'Perf': epoch_perf
    }
    self.results[task][mode].append(result_dict)
    self.save_results(task, mode)
    # Display result
    self.logger.info(f'<{self.config_idx}> {task} ({self.rnn_type[task]}) [{mode}] Epoch {epoch_idx}/{self.train_epochs}, Loss={epoch_loss:.4f}, Perf={epoch_perf:.4f}')
    # Save the model if we see new best validation loss_main
    if (mode == 'Valid') and (self.best_epoch_idx[task] == -1 or epoch_perf > self.best_valid_perf[task]):
      self.best_epoch_idx[task] = epoch_idx
      self.best_valid_perf[task] = epoch_perf
      self.save_model(task)

    return epoch_perf


  def dynamic_eval(self, mode='Dynamic'):
    # Set parameters for dynamic evaluation
    lamb = self.cfg['lamb']
    epsilon = self.cfg['epsilon']
    lr = self.cfg['optimizer']['lr'][self.main_task]

    # Load LM model for dynamic evaluation
    self.load_model(self.main_task)

    self.logger.info('Collect gradient statistics on training data ...')
    self.gradstat()
    
    self.logger.info('Start dynamic evaluation ...')
    # Set model to train mode
    self.set_model_mode('Train')
    # Deactivate dropout
    self.encoder.use_dropout = False
    for t in self.tasks:
      self.decoders[t].use_dropout = False
    '''
    Clip decay rates at 1/lamb,
    otherwise scaled decay rates can be greater than 1
    which would cause decay updates to overshoot.
    '''
    for param in self.params[self.main_task]:
      if self.device == 'cuda':
        decratenp = param.decrate.cpu().numpy()
        ind = np.nonzero(decratenp > (1/lamb))
        decratenp[ind] = (1 / lamb)
        param.decrate = torch.from_numpy(decratenp).type(torch.cuda.FloatTensor)
        param.data0 = 1 * param.data
      else:
        decratenp = param.decrate.numpy()
        ind = np.nonzero(decratenp > (1/lamb))
        decratenp[ind] = (1 / lamb)
        param.decrate = torch.from_numpy(decratenp).type(torch.FloatTensor)
        param.data0 = 1 * param.data
    # Initailize hidden states
    hiddens = {}
    for t in self.tasks:
      hiddens[t] = self.decoders[t].init_hidden(self.batch_sizes['Test'])
    
    total_loss = 0
    batch_num, location = 0, 0
    last = False
    seq_len = self.bptt_len
    seq_len0 = seq_len
    while location < self.data['Test']['text'].size(0) - 1 - 1:
      # Get last chunk of seqlence if seq_len doesn't divide full sequence cleanly
      if location + seq_len >= self.data['Test']['text'].size(0):
        if last:
          break
        seq_len = self.data['Test']['text'].size(0) - location - 1
        last = True

      text, next_text, label, next_label = get_batch(self.data['Test'], location, self.bptt_len)
      # Forward
      self.optimizers[self.main_task].zero_grad()
      # Foward
      x_emb = self.encoder(text)
      v_backward = compute_v_backward(label, self.gamma, self.task_output_sizes['label'])
      if self.aux_task is not None:
        auxiliary_input = self.decoders[self.aux_task](x_emb, hiddens[self.aux_task], v_backward)
        if self.GVF_discounts[self.aux_task] < 0:
          auxiliary_input = F.softmax(auxiliary_input, -1)
      else:
        auxiliary_input = None

      log_prob, hiddens[self.main_task], _, _ = self.decoders[self.main_task](x_emb, hiddens[self.main_task], auxiliary_input)
      
      # Get loss and backward
      loss_tensor = self.loss_funcs[self.main_task](log_prob, next_text)
      loss_tensor.backward()
      # Optimization step
      for param in self.params[self.main_task]:
        dW = lamb * param.decrate * (param.data0 - param.data) - lr * param.grad.data / (param.MS + epsilon)
        param.data += dW
      
      # seq_len / self.bptt_len will be 1 except for last sequence. For last sequence, we downweight if sequence is shorter.
      total_loss += loss_tensor.item() * seq_len / self.bptt_len
      batch_num += (seq_len / self.bptt_len)
      location += seq_len

    # Save result
    epoch_loss = total_loss / batch_num
    epoch_perf = -1.0* epoch_loss
    result_dict = {
      'Batch': -1,
      'RNN': self.rnn_type[self.main_task],
      'Task': self.main_task,
      'Loss': epoch_loss,
      'Perf': epoch_perf
    }
    self.results[self.main_task]['Dynamic'].append(result_dict)
    self.save_results(self.main_task, 'Dynamic')
    # Display result
    self.logger.info(f'<{self.config_idx}> {self.main_task} ({self.rnn_type[self.main_task]}) [{mode}] lamb={lamb:.4f}, lr={lr:.4f}, epsilon={epsilon:.6f}')
    self.logger.info(f'<{self.config_idx}> {self.main_task} ({self.rnn_type[self.main_task]}) [{mode}] Loss={epoch_loss:.4f}, PPL={np.exp(epoch_loss):.2f}')
    self.logger.info('-'*20)


  def gradstat(self, mode='Train'):
    task = self.main_task
    # Set model to train mode
    self.set_model_mode(mode)
    # Deactivate dropout
    self.encoder.use_dropout = False
    for t in self.tasks:
      self.decoders[t].use_dropout = False
    # Initailizae MS to zeros
    for param in self.params[self.main_task]:
      param.MS = 0 * param.data
    # Initailize hidden states
    hiddens = {}
    for t in self.tasks:
      hiddens[t] = self.decoders[t].init_hidden(self.batch_sizes[mode])

    total_loss = 0
    batch_num, location = 0, 0
    while location < self.data[mode]['text'].size(0) - 1 - 1:
      text, next_text, label, next_label = get_batch(self.data[mode], location, self.bptt_len)
      # Forward
      self.optimizers[self.main_task].zero_grad()
      x_emb = self.encoder(text)
      v_backward = compute_v_backward(label, self.gamma, self.task_output_sizes['label'])
      if self.aux_task is not None:
        auxiliary_input = self.decoders[self.aux_task](x_emb, hiddens[self.aux_task], v_backward)
        if self.GVF_discounts[self.aux_task] < 0:
          auxiliary_input = F.softmax(auxiliary_input, -1)
      else:
        auxiliary_input = None
      log_prob, hiddens[self.main_task], _, _ = self.decoders[self.main_task](x_emb, hiddens[self.main_task], auxiliary_input)

      # Get loss && performance measurement
      loss_tensor = self.loss_funcs[task](log_prob, next_text)
      loss_tensor.backward()
      for param in self.params[self.main_task]:
        param.MS = param.MS + param.grad.data * param.grad.data
      total_loss += loss_tensor.item()
      batch_num += 1
      location += self.bptt_len

    # update MS
    gsum = 0
    for param in self.params[self.main_task]:
      param.MS = torch.sqrt(param.MS)
      gsum += torch.mean(param.MS)
    for param in self.params[self.main_task]:
      param.decrate = param.MS / gsum


  def save_results(self, task, mode):
    results = pd.DataFrame(self.results[task][mode])
    results['Task'] = results['Task'].astype('category')
    results['RNN'] = results['RNN'].astype('category')
    results.to_feather(self.log_paths(task, mode))

  def save_config(self):
    cfg_json = json.dumps(self.cfg, indent=2)
    f = open(self.cfg['cfg_path'], 'w')
    f.write(cfg_json)
    f.close()

  def save_model(self, task):
    model_dict_list = [self.encoder.state_dict()]
    if task == self.main_task:
      for t in self.tasks:
        model_dict_list.append(self.decoders[t].state_dict())
    else:
      model_dict_list.append(self.decoders[task].state_dict())
    torch.save(model_dict_list, self.model_path(task))

  def load_model(self, task):
    model_dict_list = torch.load(self.model_path(task))
    # Load encdoer
    self.encoder.load_state_dict(model_dict_list[0])
    self.encoder = self.encoder.to(self.device)
    # Load decoders
    i = 1
    if task == self.main_task:
      for t in self.tasks:
        self.decoders[t].load_state_dict(model_dict_list[i])
        self.decoders[t] = self.decoders[t].to(self.device)
        i += 1
    else:
      self.decoders[task].load_state_dict(model_dict_list[i])
      self.decoders[task] = self.decoders[task].to(self.device)

  def save_embedding(self, task):
    self.load_model(task)
    embedding_matrix = self.encoder.encoder.weight.numpy()
    self.logger.info(f'Save embedding matrix: {embedding_matrix.size()}')

  def set_model_mode(self, mode):
    if mode == 'Train':
      self.encoder.train()
      for task in self.tasks:
        self.decoders[task].train()
    else:
      self.encoder.eval()
      for task in self.tasks:
        self.decoders[task].eval()