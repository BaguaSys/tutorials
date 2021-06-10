# Run

Bagua has a built-in tool `bagua.distributed.launch` to launch jobs, whose usage is similar to Pytorch `torch.distributed.launch`.

We introduce how to start distributed training in the following sections.

## Single node multi-process training

```shell
$ python -m bagua.distributed.launch --nproc_per_node=8 \
  your_training_script.py (--arg1 --arg2 ...)
```

## Multi-node multi-process training (e.g. two nodes)

**Node 1**: *(IP: 192.168.1.1, and has a free port: 1234)*
```shell
$ python -m bagua.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=1234  your_training_script.py (--arg1 --arg2 ...)
```

**Node 2**:
```shell
$ python -m bagua.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="192.168.1.1" --master_port=1234 your_training_script.py (--arg1 --arg2 ...)
```

> *Tips*:
>
> If you need some preprocessing work, you can include them in your bash script and launch job by adding `--no_python` to your command.
> ``` shell
> python -m bagua.distributed.launch --no_python --nproc_per_node=8 bash your_bash_script.sh
> ```
