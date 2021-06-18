# Elastic Training

## Introduction

By applying [TorchElastic](https://github.com/pytorch/pytorch/tree/v1.9.0/torch/distributed/elastic), bagua can do elastic training. We usually use the capabilities of Elastic Training to solve the following two problems:

**Fault Tolerant Jobs**

Jobs that run on infrastructure where nodes get replaced frequently, either due to flaky hardware or by design. Or mission critical production grade jobs that need to be run with resilience to failures.

**Dynamic Capacity Management**

Jobs that run on leased capacity that can be taken away at any time (e.g. AWS spot instances) or shared pools where the pool size can change dynamically based on demand.

## Quickstart

### 1. Make your program recoverable

Elastic training means that new nodes will be added during the training process. Your training program need to save the training status in time, so that the new joining process can join the training from the most recent state.

For example:

```python
model = ...
model.load_state_dict(torch.load(YOUR_CHECKPOINT_PATH))

for train_loop():
    ...
    torch.save(model.state_dict(), YOUR_CHECKPOINT_PATH)
```

### 2. Launch job

You can launch Elastic Training job with `bagua.distributed.run`. For example:

**Fault tolerant (fixed sized number of workers, no elasticity)**

```bash
python -m bagua.distributed.run \
        --nnodes=$NUM_NODES \
        --nproc_per_node=$NUM_TRAINERS \
        --rdzv_id=$JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$HOST_NODE_ADDR \
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

```

Part of the node failure will not cause the job to fail, the job will wait for the node to recover.

``HOST_NODE_ADDR``, in form <host>[:<port>] (e.g. node1.example.com:29400), specifies the node and
the port on which the C10d rendezvous backend should be instantiated and hosted. It can be any
node in your training cluster, but ideally you should pick a node that has a high bandwidth.

> If no port number is specified ``HOST_NODE_ADDR`` defaults to 29400.


**Elastic Training(min=1, max=4)**

```bash
python -m bagua.distributed.run \
        --nnodes=1:4 \
        --nproc_per_node=$NUM_TRAINERS \
        --rdzv_id=$JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$HOST_NODE_ADDR \
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

```

For this example, the training node can be dynamically adjusted from 1 to 4.

## Reference

1. [PYTORCH ELASTIC overview](https://pytorch.org/elastic/0.1.0rc2/overview.html)
2. [torch.distributed.run API Doc](https://github.com/pytorch/pytorch/blob/v1.9.0/torch/distributed/run.py)
