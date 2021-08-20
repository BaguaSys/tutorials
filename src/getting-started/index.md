# Getting Started

Using Bagua is similar to using other distributed training libraries like PyTorch DDP and Horovod.

## Migrate from your existing single GPU training code

To use `Bagua`, you need make the following changes on your training code:

First, import bagua:

```python
import bagua.torch_api as bagua
```

Then initialize Bagua's process group:

```python
torch.cuda.set_device(bagua.get_local_rank())
bagua.init_process_group()
```

Then, use torch's distributed sampler for your data loader:

```python
train_dataset = ...
test_dataset = ...

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
    num_replicas=bagua.get_world_size(), rank=bagua.get_rank())

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
)

test_loader = torch.utils.data.DataLoader(test_dataset, ...)
```

Finally, wrap you model and optimizer with bagua by adding one line of code to your original script:

```python
def main():
    args = parse_args()
    # define your model and optimizer
    model = MyNet().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # transform to Bagua wrapper
    from bagua.torch_api.algorithms import gradient_allreduce
    model = model.with_bagua(
        [optimizer], gradient_allreduce.GradientAllReduceAlgorithm()
    )

    # train the model over the dataset
    for epoch in rage(args.epochs):
        for b_idx, (inputs, targets) in enumerate(train_loader):
            outputs = model(inputs)
            loss = torch.nn.CrossEntropyLoss(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

More examples can be found [here](https://github.com/BaguaSys/examples).

## Launch job

Bagua has a built-in tool `bagua.distributed.launch` to launch jobs, whose usage is similar to Pytorch `torch.distributed.launch`.

We introduce how to start distributed training in the following sections.

### Single node multi-process training

```shell
python -m bagua.distributed.launch --nproc_per_node=8 \
  your_training_script.py (--arg1 --arg2 ...)
```

### Multi-node multi-process training (e.g. two nodes)

#### Method 1: run command on each node

**Node 1**: *(IP: 192.168.1.1, and has a free port: 1234)*
```shell
python -m bagua.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=1234  your_training_script.py (--arg1 --arg2 ...)
```

**Node 2**:
```shell
python -m bagua.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="192.168.1.1" --master_port=1234 your_training_script.py (--arg1 --arg2 ...)
```

> *Tips*:
>
> If you need some preprocessing work, you can include them in your bash script and launch job by adding `--no_python` to your command.
> ``` shell
> python -m bagua.distributed.launch --no_python --nproc_per_node=8 bash your_bash_script.sh
> ```

#### Method 2: run command on a single node providing a host list

If the ssh service is available with passwordless login on each node, we can launch a distributed job on a single node with a similar syntax as `mpirun`.

Bagua provides a program `baguarun`. For the multi-node training example above, the command to start with `bagurun` looks like:

```shell
baguarun --host_list NODE_1_HOST:NODE_1_SSH_PORT,NODE_2_HOST:NODE_2_SSH_PORT \
                --nproc_per_node=NUM_GPUS_YOU_HAVE --master_port=1234 \
                YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other arguments of your training script)
```
