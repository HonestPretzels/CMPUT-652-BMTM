# PRL

This repository contains the code used for paper "Predictive Representation Learning for Sequence Labeling". The new model Predictive Representation Learning (PRL) combine [General Value Function Networks (GVFNs)](https://arxiv.org/abs/1807.06763) with [MoS](https://github.com/zihangdai/mos).

## Requirements

- PyTorch: You may change the version of cuda toolkit according to your GPU version.

  ```conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch```

- Others:

  ```pip install -r requirements.txt```


## Experiments

### Training

All hyperparameters including parameters for grid search are stored in a json file in directory `configs`. To run for an experiment, a config index is first given to generate a config dict corresponding to this specific config index. Then we run the experiment specified by this config dict. All results including log files and model file are saved in directory `logs`. Please refer the code for details.

For each task in config file (e.g. PennTreebank, PTB_POS), there is a discount factor called GVF_dicount. When GVF_dicount<0, we don't apply GVFN for this task and the decoder outputs a probability distribution (e.g. MoS-PRL-P); otherwise, we apply GVFN for this task with a discount factor equals to GVF_dicount and the decoder outputs the action-values (e.g. MoS-PRL-Q).

`gamma` is used to compute the label trace. When `gamma>0`, we use the label trace component defined with `gamma`; when `gamma<0`, we remove the labal trace.

To run the experiment with config file `ptb_best.json` and config index `1`:

- train, validate, and test:

  ```python main.py --mode Train --config_file ./configs/ptb_best.json --config_idx 1```

- test without dynamic evaluation:
  
  ```python main.py --mode Test --config_file ./configs/ptb_best.json --config_idx 1```

- test with dynamic evaluation:

  ```python main.py --mode Dynamic --lamb 0.075 --config_file ./configs/ptb_best.json --config_idx  1```

### Reproduce the results in the paper
  - Hyperparameter tuning: we apply grid search to select the best parameter configuration from `ptb_pos_search.json` or `wt2_pos_search.json`.
    ```
    for index in {1..2340}
    do
      python main.py --mode Train --config_file ./configs/ptb_pos_search.json --config_idx $index
      python main.py --mode Train --config_file ./configs/wt2_pos_search.json --config_idx $index
    done
    ```
  
  - Train, Validate and test MoS-PRL-Q with the best configuration for 5 runs, without dynamic evaluation:
    ```
    for index in {1..5}
    do
      python main.py --mode Train --config_file ./configs/ptb_pos_best.json --config_idx $index
      python main.py --mode Train --config_file ./configs/wt2_pos_best.json --config_idx $index
    done
    ```

  - Test MoS-PRL-Q with the best configuration for 5 runs, with dynamic evaluation:
    ```
    for index in {1..5}
    do
      python main.py --mode Dynamic --lamb 0.075 --config_file ./configs/ptb_pos_best.json --config_idx $index
      python main.py --mode Dynamic --lamb 0.02 --lr 0.002 --epsilon 0.006 --config_file ./configs/wiki2_pos_best.json --config_idx $index
    done
    ```
  

# Acknowledgements
- MoS: https://github.com/zihangdai/mos
- awd-lstm-lm: https://github.com/salesforce/awd-lstm-lm
- Dynamic evaluation: https://github.com/benkrause/dynamic-evaluation