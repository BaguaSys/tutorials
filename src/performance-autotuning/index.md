---
modified: 2021-07-01T06:38:23.018Z
title: Performance Autotuning
---

# Performance Autotuning

Bagua comes with several adjustable hyperparameters that can affect runtime performance, including tensor fusion bucket size and hierarchical collective algorithms.

Determining the best combination of these values to maximize performance (minimize time to convergence) can be a matter of trial-and-error, as many factors including model complexity, network bandwidth, GPU memory, etc. can all affect inputs per second throughput during training.

Bagua provides a mechanism to automate the process of selecting the best values for these hyperparameters called autotuning. The Bagua autotuning system uses Bayesian optimization to intelligently search through the space of parameter combinations during training. This feature can be enabled by setting the `--autotune_level=1` flag for `bagua.distributed.run`:

```bash
python -m bagua.distributed.run --nproc_per_node 4 --auotune_level=1 python train.py
```

## Autotune processing

The main process of autotune is simple. Autotuning system find N groups of hyperparameters through Bayesian, and the hyperparameters are brought into the training to verify the effect, each hyperparameter takes T seconds to verify. 

Generally speaking, the larger N is, the larger the search space, and the more likely it is to find the parameters you want. The larger the T, the more accurate the scoring.

In addition, due to the cold start, we will skip the sampling in the previous W seconds to prevent sampling distortion.

You can adjust N through the `--autotune_max_samples` flag and adjust T through the `--autotune_sampling_confidence_time` flag, adjust W through the `--autotune_warmup_time`.

## Autotune logfile

The autotuning system dumps the tuning process in a file. The file path is specified by the parameter `--autotune_logfile`, and the default value is `/tmp/bagua_autotune.log`.

The file format is csv, each row is the hyperparameters and scoring of a round of iteration:

```csv
bucket_size_2p,is_hierarchical_reduce,score,train_iter
23,False,1.4117491757441083,300
15,True,1.1258082798810858,400
29,True,1.0463204022477832,500
```

`bucket_size_2p` is the power of 2 of the bucket size, for example `bucket_size_2p`=23 means bucket_size is 8388608 bytes (2 ^ 23).
