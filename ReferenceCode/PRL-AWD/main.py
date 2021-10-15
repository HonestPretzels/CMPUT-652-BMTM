import os
import sys
import json
import argparse

from utils.sweeper import Sweeper
from experiment import Experiment
from utils.helper import make_dir

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def main(argv):
  parser = argparse.ArgumentParser(description="Config file")
  parser.add_argument('--config_file', type=str, default='./configs/gvfn.json', help='Configuration sweep file')
  parser.add_argument('--config_idx', type=int, default=1, help='Configuration index')
  parser.add_argument('--mode', type=str, default='Train', help='Mode: [Train, Test, Neural]')
  parser.add_argument('--slurm_dir', type=str, default='', help='slurm tempory directory')
  # For neural cache
  parser.add_argument('--theta', type=float, default=1.0, help='mix rate of uniform distribution and pointer softmax distribution')
  parser.add_argument('--window', type=int, default=500, help='pointer window length')
  parser.add_argument('--lambdasm', type=float, default=0.1, help='linear mix rate')
  parser.add_argument('--bptt_len', type=int, default=5000, help='bptt length')

  args = parser.parse_args()

  if args.mode == 'Neural':
    exp = args.config_file.split('/')[-1].split('.')[0]
    cfg_path = f'./logs/{exp}/{args.config_idx}/config.json'
    with open(cfg_path, 'r') as f:
      cfg = json.load(f)
    assert cfg['tasks'][-1] in ['PennTreebank', 'WikiText2', 'WikiText103'], 'The last task must be a Language Modeling task.'
    cfg['logs_dir'] = f"./logs/{cfg['exp']}/{cfg['config_idx']}/"
    # Set parameters for neural cache
    cfg['theta'] = args.theta
    cfg['window'] = args.window 
    cfg['lambdasm'] = args.lambdasm
    cfg['bptt_len'] = args.bptt_len
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
  # Set
  cfg['mode'] = args.mode  
  # Run
  exp = Experiment(cfg)
  exp.run()

if __name__=='__main__':
  main(sys.argv)