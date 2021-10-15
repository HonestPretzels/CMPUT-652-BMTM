import os
import sys
import json
import argparse

from utils.sweeper import Sweeper
from experiment import Experiment
from utils.helper import make_dir

def main(argv):
  parser = argparse.ArgumentParser(description="Config file")
  parser.add_argument('--config_file', type=str, default='./configs/gvfn.json', help='Configuration sweep file')
  parser.add_argument('--config_idx', type=int, default=1, help='Configuration index')
  parser.add_argument('--mode', type=str, default='Train', help='Mode: [Train, Test, Dynamic]')
  parser.add_argument('--slurm_dir', type=str, default='', help='slurm tempory directory')
  # For dynamic evaluation
  parser.add_argument('--lamb', type=float, default=0.02, help='decay parameter lambda')
  parser.add_argument('--epsilon', type=float, default=0.001, help='stabilization parameter epsilon')
  parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
  parser.add_argument('--batch_size', type=int, default=100, help='batch size for gradient statistics')
  parser.add_argument('--bptt_len', type=int, default=5, help='bptt length')
  args = parser.parse_args()

  if args.mode == 'Dynamic':
    exp = args.config_file.split('/')[-1].split('.')[0]
    cfg_path = f'./logs/{exp}/{args.config_idx}/config.json'
    with open(cfg_path, 'r') as f:
      cfg = json.load(f)
    assert cfg['tasks'][-1] in ['PennTreebank', 'WikiText2', 'WikiText103'], 'The last task must be a Language Modeling task in dynamic evaluation.'
    # Set parameters for dynamic evaluation
    cfg['lamb'] = args.lamb
    cfg['epsilon'] = args.epsilon
    cfg['optimizer']['lr'][cfg['tasks'][-1]] = args.lr
    cfg['batch_size'] = args.batch_size
    cfg['bptt_len'] = args.bptt_len
    cfg['logs_dir'] = f"./logs/{cfg['exp']}/{cfg['config_idx']}/"
  else:
    sweeper = Sweeper(args.config_file)
    cfg = sweeper.generate_config_for_idx(args.config_idx)
    # Set experiment name and log paths 
    cfg['exp'] = args.config_file.split('/')[-1].split('.')[0]
    if len(args.slurm_dir) > 0:  
      cfg['logs_dir'] = f"{args.slurm_dir}/{cfg['exp']}/{cfg['config_idx']}/"
      make_dir(cfg['logs_dir'])
    else:
      cfg['logs_dir'] = f"./logs/{cfg['exp']}/{cfg['config_idx']}/"
    make_dir(f"./logs/{cfg['exp']}/{cfg['config_idx']}/")
    cfg_path = cfg['logs_dir'] + 'config.json'
    cfg['cfg_path'] = cfg_path
    # Set tasks
    cfg['tasks'] = [task for task in cfg['GVF_discounts']]
    for i in range(len(cfg['tasks'])):
      if cfg['tasks'][i] in ['PennTreebank', 'WikiText2', 'WikiText103']:
        cfg['tasks'][-1], cfg['tasks'][i] = cfg['tasks'][i], cfg['tasks'][-1]
        break

  # Set mode
  cfg['mode'] = args.mode
  # Run
  exp = Experiment(cfg)
  exp.run()

if __name__=='__main__':
  main(sys.argv)