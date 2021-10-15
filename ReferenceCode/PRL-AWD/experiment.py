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
    if self.cfg['generate_random_seed'] and self.cfg['mode'] != 'Neural':
      self.cfg['seed'] = np.random.randint(int(1e6))
    
    self.results = {}
    for task in cfg['tasks']:
      self.results[task] = {}
      for mode in ['Train', 'Test', 'Valid']:
        self.results[task][mode] = []

    self.log_paths = lambda task, mode: cfg['logs_dir'] + f'result_{task}_{mode}.feather'
    self.model_path = lambda task: cfg['logs_dir'] + f'model_{task}.pt'

    if self.cfg['mode'] == 'Neural':
      self.logger = Logger(cfg['logs_dir'], file_name='log_neural.txt', filemode='a')
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
    self.batch_sizes = {
      'Train': cfg['batch_size'],
      'Valid': 10,
      'Test': 1
    }
    self.train_epochs = cfg['train_epochs']
    self.mode = cfg['mode']
    self.bptt_len = cfg['bptt_len']
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
      self.loss_funcs[task] = get_loss_func(task, self.GVFs[task], self.GVF_discounts[task], self.cfg['embedding']['size'])
      self.perf_funcs[task] = get_perf_func(task, self.GVFs[task], self.GVF_discounts[task])
      if task == self.main_task:
        self.logger.info('Build Model: LM Decoder ...')
        self.decoders[task] = LMDecoder(self.cfg, self.encoder.encoder, self.task_output_sizes, self.layer_select).to(self.device)
        self.params[task] = list(self.decoders[task].parameters())
      else:
        aux_cfg[self.main_task]['gamma'] = self.cfg['gamma']
        aux_cfg[self.main_task]['mlp'] = self.cfg['mlp']
        aux_cfg[self.main_task]['embedding'] = self.cfg['embedding']
        aux_cfg[self.main_task]['hidden_activation'] = self.cfg['hidden_activation']
        aux_cfg[self.main_task]['output_activation'] = self.cfg['output_activation']
        self.logger.info('Build Model: Auxiliary Decoder ...')
        self.decoders[task] = AuxDecoder(aux_cfg[self.main_task], self.task_output_sizes['label']).to(self.device)
        self.params[task] = list(self.encoder.parameters()) + list(self.decoders[task].parameters())
      model_size += count_params(self.decoders[task])/1e6
      self.optimizers[task] = getattr(torch.optim, self.cfg['optimizer']['name'])(self.params[task], lr=self.cfg['optimizer']['lr'][task], weight_decay=self.cfg['optimizer']['weight_decay'])
    self.logger.info(f'Model size: {model_size:.2f} M')   

    self.logger.info('Start running ...')
    if self.mode == 'Train': # Train && Valid
      self.train()
    if self.mode in ['Train', 'Test']: # Test
      for task in self.tasks:
        _ = self.evaluate('Test', task, self.best_epoch_idx[task])
    if self.mode == 'Neural':
      self.neural_cache()
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
    
    # Start training
    for i in range(1, self.train_epochs+1):
      # Initailization hidden states
      self.hiddens = {}
      for task in self.tasks:
        self.hiddens[task] = self.decoders[task].init_hidden(self.batch_sizes[mode])
      self.set_model_mode(mode)
      location = 0
      epoch_loss, epoch_perf = {}, {}
      for task in self.tasks:
        epoch_loss[task], epoch_perf[task] = 0.0, 0.0
      while location < self.data[mode]['text'].size(0) - 1 - 1:
        # Get sequence length
        bptt_len = self.bptt_len if np.random.random() < 0.95 else self.bptt_len / 2.0
        seq_len = max(5, int(np.random.normal(bptt_len, 5))) # Prevent a very small sequence length
        # Change lr
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

    # ------------> main task <------------
    task = self.main_task
    v_backward = compute_v_backward(label, self.gamma, self.task_output_sizes['label'])
    # Foward
    x_emb = self.encoder(text)
    if self.aux_task is not None:
      auxiliary_input = self.decoders[self.aux_task](x_emb, self.hiddens[self.aux_task], v_backward)
      if self.GVF_discounts[self.aux_task] < 0:
        auxiliary_input = F.softmax(auxiliary_input, -1)
    else:
      auxiliary_input = None
    result, self.hiddens[task], rnn_hs, dropped_rnn_hs = self.decoders[task](x_emb, self.hiddens[task], auxiliary_input)
    # Get loss
    loss_tensor = self.loss_funcs[task](self.decoders[task].decoder.weight, self.decoders[task].decoder.bias, result, next_text.view(-1))
    loss[task] = loss_tensor.item()
    # Activiation Regularization
    if self.cfg['optimizer']['alpha']>0:
      loss_tensor += sum(self.cfg['optimizer']['alpha'] * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
    # Temporal Activation Regularization
    if self.cfg['optimizer']['beta']>0:
      loss_tensor += sum(self.cfg['optimizer']['beta'] * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
    # Backward
    self.optimizers[task].zero_grad()
    loss_tensor.backward()
    # Get performance measurement
    perf[task] = -1.0 * loss[task]
    # Take a optimization step
    if self.cfg['optimizer']['gradient_clip']>0: # Clip the gradients
      torch.nn.utils.clip_grad_norm_(self.params[task], self.cfg['optimizer']['gradient_clip'])
    self.optimizers[task].step()
   
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
      self.optimizers[task].zero_grad()
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
          result, eval_hidden[task], _, _ = self.decoders[task](x_emb, eval_hidden[task], auxiliary_input)
          # Get loss && perf
          loss = self.loss_funcs[task](self.decoders[task].decoder.weight, self.decoders[task].decoder.bias, result, next_text.view(-1)).item()
          perf = -1.0*loss
        else:
          loss = self.loss_funcs[task](state_value, next_label).item()
          perf = self.perf_funcs[task](state_value, next_label)
        # Add epoch loss && perf
        epoch_loss += loss * len(text)
        epoch_perf += perf * len(text)
    
    # Save result
    total_seq_len = len(self.data[mode]['text'])
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


  def neural_cache(self, mode='Test'):
    task = self.main_task
    theta = self.cfg['theta']
    window = self.cfg['window']
    lambdasm = self.cfg['lambdasm']
    batch_size = self.batch_sizes[mode]
    
    # Load LM model for neural cache
    self.load_model(task)
    self.logger.info('Start neural cache ...')
    # Set model to test mode
    self.set_model_mode(mode)
    # Initailization evaluation hidden states
    neural_hidden = {}
    for t in self.tasks:
      neural_hidden[t] = self.decoders[t].init_hidden(self.batch_sizes[mode])
    
    # Start neural cache
    epoch_loss = 0
    next_word_history = None
    pointer_history = None
    with torch.no_grad():
      for location in range(0, self.data[mode]['text'].size(0) - 1, self.bptt_len):
        text, next_text, label, next_label = get_batch(self.data[mode], location, self.bptt_len)
        targets = next_text.view(-1)
        # Prevent excessively small sequence length or wrong batch size
        if text.size(0) < 3 or text.size(1) != self.batch_sizes[mode]:
          continue
        # Foward
        x_emb = self.encoder(text)
        v_backward = compute_v_backward(label, self.gamma, self.task_output_sizes['label'])
        if self.aux_task is not None:
          state_value = self.decoders[self.aux_task](x_emb, neural_hidden[self.aux_task], v_backward)
        if self.aux_task is not None:
          auxiliary_input = F.softmax(state_value, -1) if self.GVF_discounts[self.aux_task] < 0 else state_value
        else:
          auxiliary_input = None
        result, neural_hidden[task], rnn_hs, _ = self.decoders[task](x_emb, neural_hidden[task], auxiliary_input)
        log_prob = self.decoders[task].decoder(result).view(-1, self.task_output_sizes['word'])
        # Fill pointer history
        rnn_out = rnn_hs[-1].squeeze()
        start_idx = len(next_word_history) if next_word_history is not None else 0
        if next_word_history is None:
          next_word_history = torch.cat([one_hot(t.item(), self.task_output_sizes['word'], self.device) for t in targets])
        else:
          next_word_history = torch.cat([next_word_history, torch.cat([one_hot(t.item(), self.task_output_sizes['word'], self.device) for t in targets])])      
        if pointer_history is None:
          pointer_history = rnn_out.clone().detach()
        else:
          pointer_history = torch.cat([pointer_history, rnn_out.clone().detach()], dim=0)
        # Pointer manual cross entropy
        loss = 0
        softmax_output_flat = F.softmax(log_prob)
        for idx, vocab_loss in enumerate(softmax_output_flat):
          p = vocab_loss
          if start_idx + idx > window:
            valid_next_word = next_word_history[start_idx+idx-window : start_idx+idx]
            valid_pointer_history = pointer_history[start_idx+idx-window : start_idx+idx]
            logits = torch.mv(valid_pointer_history, rnn_out[idx])
            ptr_attn = F.softmax(theta * logits).view(-1, 1)
            ptr_dist = (ptr_attn.expand_as(valid_next_word) * valid_next_word).sum(0).squeeze()
            p = lambdasm * ptr_dist + (1 - lambdasm) * vocab_loss
          target_loss = p[targets[idx].item()]
          loss += (-torch.log(target_loss)).item()
        next_word_history = next_word_history[-window:]
        pointer_history = pointer_history[-window:]
        # Get loss
        epoch_loss += loss / batch_size

    epoch_loss = epoch_loss / self.data[mode]['text'].size(0)
    # Display result
    self.logger.info(f'<{self.config_idx}> {self.main_task} ({self.rnn_type[self.main_task]}) [Neural] Loss={epoch_loss:.4f}, PPL={np.exp(epoch_loss):.2f}')
    self.logger.info('-'*20)


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


  def set_model_mode(self, mode):
    if mode == 'Train':
      self.encoder.train()
      for task in self.tasks:
        self.decoders[task].train()
    else:
      self.encoder.eval()
      for task in self.tasks:
        self.decoders[task].eval()