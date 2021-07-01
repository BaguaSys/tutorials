# ByteGrad

## Overview

Large scale distributed training requires significant communication cost, which
is especially true for large models. For example in traditional synchronous
distributed setup with AllReduce to synchronize gradients (which is the case for
many libraries, such as [Horovod](https://github.com/horovod/horovod) and
[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)), in
each iteration of the training process, the gradient, whose size is equal to the
model size, needs to be sent and received on each worker. Such communication
cost soon becomes the training bottleneck in many scenarios.

There are many [existing
papers](https://awesomeopensource.com/project/chester256/Model-Compression-Papers?categoryPage=21)
about how to apply model/gradient compression to save this communication cost.
Bagua provides a built-in gradient compression algorithm called **ByteGrad**,
which compresses the gradient floats to 8bit bytes before communication. This
saves 3/4 of the original cost. It implements high accuracy min-max quantization
operator with optimized CUDA kernels, and hierarchical communication. This makes
it much faster than other compression implementations in existing frameworks
(such as [PyTorch
PowerSGD](https://pytorch.org/docs/stable/ddp_comm_hooks.html#powersgd-communication-hook))
and converges similar to full precision communication on most tasks:

benchmark results XXX

For comparison with other algorithms (may or may not be compression algorithms),
refer to [benchmark page](../benchmark/index.html).

## Algorithm

ByteGrad does the following steps in each iteration. Assume we have $m$ nodes
and each node has $n$ GPUs.

1. Calculate gradient $g_{ij}$ on the $i$-th node's $j$-th GPU for all $i,j$
2. The first GPU on each *node* does a reduce operation to compute the average
   of all GPUs' gradients *on the same node*, defined as $G_i$ for the $i$-th
   node
3. The first GPU on $i$-th node quantize the gradient $G_i$ with a quantization
   function $Q(\cdot)$: $Q(G_i)$, for all $i$. Then each node exchange the
   quantized version between nodes so that each node has the average of all
   $Q(G_i)$
4. The first GPU on each node broadcast the average of all $Q(G_i)$s to every
   other GPU on the same node, and all GPUs on all workers use this quantized
   average to update model
   
The quantization function $Q(\cdot)$ calculates the minimum value $x$ and
maximum value $y$ of its input, and the split $[x, y]$ into evenly spaced 256
intervals. Then represent each element of its input by a 8bit integer
representing which interval the original element is in.

## Example usage

A complete example of running ByteGrad can be found at [Bagua examples](https://github.com/BaguaSys/examples/blob/main/benchmark/synthetic_benchmark.py)
with `--algorithm bytegrad` command line argument.

You need to initialize the Bagua algorithm with (see [API documentation](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/algorithms/bytegrad/index.html) for what parameters you can customize):

```python
from bagua.torch_api.algorithms import bytegrad
algorithm = bytegrad.ByteGradAlgorithm()
```

Then decorate your model with:

```python
model = model.with_bagua([optimizer], algorithm)
```
