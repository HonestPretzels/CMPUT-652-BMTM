# PRL

This repository contains the code used for paper "Predictive Representation Learning for Language Modeling". The new model Predictive Representation Learning (PRL) combine [General Value Function Networks (GVFNs)](https://arxiv.org/abs/1807.06763) with [AWD](https://github.com/salesforce/awd-lstm-lm).


## Requirements

- PyTorch: You may change the version of cuda toolkit according to your GPU version.

  ```conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch```

- Others:

  ```pip install -r requirements.txt```


## Experiments

### Training

All hyperparameters including parameters for grid search are stored in a json file in directory `configs`. To run for an experiment, a config index is first given to generate a config dict corresponding to this specific config index. Then we run the experiment specified by this config dict. All results including log files and model file are saved in directory `logs`. Please refer the code for details.

For each task in config file (e.g. PennTreebank, PTB_POS), there is a discount factor called GVF_dicount. When GVF_dicount<0, we don't apply GVFN for this task and the decoder outputs a probability distribution (e.g. AWD-PRL-P); otherwise, we apply GVFN for this task with a discount factor equals to GVF_dicount and the decoder outputs the action-values (e.g. AWD-PRL-Q).

`gamma` is used to compute the label trace. When `gamma>0`, we use the label trace component defined with `gamma`; when `gamma<0`, we remove the labal trace.

To run the experiment with config file `ptb_best.json` and config index `1`:

- train, validate, and test:

  ```python main.py --mode Train --config_file ./configs/ptb_best.json --config_idx 1```

- test without neural cache:
  
  ```python main.py --mode Test --config_file ./configs/ptb_best.json --config_idx 1```

- test with neural cache:

  ```python main.py --mode Neural --config_file ./configs/ptb_best.json --lambdasm 0.1 --theta 1.0 --window 500 --bptt_len 5000 --config_idx 1```

### Reproduce the results in the paper
  - Hyperparameter tuning: we apply grid search to select the best parameter configuration from `ptb_pos_search.json` or `wt2_pos_search.json`.
    ```
    for index in {1..637}
    do
      python main.py --mode Train --config_file ./configs/ptb_pos_search.json --config_idx $index
      python main.py --mode Train --config_file ./configs/wt2_pos_search.json --config_idx $index
    done
    ```
  
  - Train, Validate and test AWD-PRL-Q and AWD-PRL-P with the best configuration for 5 runs, without neural cache:
    ```
    for index in {1..10}
    do
      python main.py --mode Train --config_file ./configs/ptb_pos_best.json --config_idx $index
      python main.py --mode Train --config_file ./configs/wt2_pos_best.json --config_idx $index
    done
    ```

  - Test AWD-PRL-Q and AWD-PRL-P with the best configuration for 5 runs, with neural cache:
    ```
    for index in {1..10}
    do
      python main.py --mode Neural --config_file ./configs/ptb_pos_best.json --lambdasm 0.1 --theta 1.0 --window 500 --bptt_len 5000 --config_idx $index
      python main.py --mode Neural --config_file ./configs/wt2_pos_best.json --lambdasm 0.1279 --theta 0.662 --window 3785 --bptt_len 2000 --config_idx $index
    done
    ```

# Acknowledgements
- awd-lstm-lm: https://github.com/salesforce/awd-lstm-lm