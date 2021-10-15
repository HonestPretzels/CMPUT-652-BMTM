import os
import copy
import torch
from collections import Counter


def load_data(dataset, batch_sizes, device):
  data_paths = {
    'PTB_POS': './data/penn_pos',
    'Wiki2_POS': './data/wiki2_pos'

  }
  corpus = Corpus(data_paths[dataset])
  all_data = {
    'Train': batchify(corpus.train, batch_sizes['Train'], device),
    'Valid': batchify(corpus.valid, batch_sizes['Valid'], device),
    'Test': batchify(corpus.test, batch_sizes['Test'], device)
  }
  dataset_output_sizes = {
    'word': len(corpus.word_dict),
    'label': len(corpus.label_dict)
  }
  '''
  # Save dictionary
  with open(f'{data_paths[dataset]}/word_dict.txt', 'w') as f:
    for i in range(len(corpus.word_dict)):
      f.write(f'{corpus.word_dict.idx2word[i]}: {i}\n')
  with open(f'{data_paths[dataset]}/label_dict.txt', 'w') as f:
    for i in range(len(corpus.label_dict)):
      f.write(f'{corpus.label_dict.idx2word[i]}: {i}\n')
  '''
  return all_data, dataset_output_sizes


def batchify(data_dict, bsz, device):
  # Work out how cleanly we can divide the dataset into bsz parts.
  nbatch = data_dict['text'].size(0) // bsz
  # Trim off any extra elements that wouldn't cleanly fit (remainders).
  data_dict['text'] = data_dict['text'].narrow(0, 0, nbatch * bsz)
  data_dict['label'] = data_dict['label'].narrow(0, 0, nbatch * bsz)
  # Evenly divide the data across the bsz batches.
  data_dict['text'] = data_dict['text'].view(bsz, -1).t().contiguous().to(device)
  data_dict['label'] = data_dict['label'].view(bsz, -1).t().contiguous().to(device)
  return data_dict


def get_batch(data_dict, i, bptt_len, seq_len=None):
  seq_len = min(seq_len if seq_len else bptt_len, len(data_dict['text']) - 1 - i)
  text = data_dict['text'][i:i+seq_len]
  next_text = data_dict['text'][i+1:i+1+seq_len]
  label = data_dict['label'][i:i+seq_len]
  next_label = data_dict['label'][i+1:i+1+seq_len]
  return text, next_text, label, next_label


class Dictionary(object):
  def __init__(self):
    self.word2idx = {}
    self.idx2word = []
    self.counter = Counter()
    self.total = 0

  def add_word(self, word):
    if word not in self.word2idx:
      self.idx2word.append(word)
      self.word2idx[word] = len(self.idx2word) - 1
    token_id = self.word2idx[word]
    self.counter[token_id] += 1
    self.total += 1
    return self.word2idx[word]

  def __len__(self):
    return len(self.idx2word)


class Corpus(object):
  def __init__(self, path):
    self.path = path
    self.word_dict = Dictionary()
    self.label_dict = Dictionary()
    self.train = self.tokenize(os.path.join(self.path, 'train_comb.txt'))
    self.valid = self.tokenize(os.path.join(self.path, 'valid_comb.txt'))
    self.test = self.tokenize(os.path.join(self.path, 'test_comb.txt'))

  def tokenize(self, path):
    assert os.path.exists(path)
    # Add words to the dictionary
    
    with open(path, 'r') as f:
      for line in f:
        l = line.strip().split()
        if len(l) == 2:
          self.word_dict.add_word(l[0])
          self.label_dict.add_word(l[1])
        else:
          self.word_dict.add_word('<eos>')
          self.label_dict.add_word('<eos>')
    # Tokenize file content
    text_ids, label_ids = [], []
    with open(path, 'r') as f:
      for line in f:
        l = line.strip().split()
        if len(l) > 0:
          assert len(l) == 2
          text_ids.append(self.word_dict.word2idx[l[0]])
          label_ids.append(self.label_dict.word2idx[l[1]])
        else:
          text_ids.append(self.word_dict.word2idx['<eos>'])
          label_ids.append(self.label_dict.word2idx['<eos>'])
    data_dict = {
      'text': torch.tensor(text_ids, dtype=torch.long),
      'label': torch.tensor(label_ids, dtype=torch.long)
    }
    return data_dict