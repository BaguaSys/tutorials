# Gradient AllReduce

The Gradient AllReduce algorithm is a popular synchronous data-parallel
distributed algorithm. It is the algorithm implemented in most existing
solutions such as [PyTorch
DistributedDataParallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html),
[Horovod](https://horovod.ai/), and [TensorFlow Mirrored
Strategy](https://www.tensorflow.org/guide/distributed_training).

With this algorithm, each worker does the following steps in each iteration.

1. Compute the gradient using a minibatch.
2. Compute the mean of the gradients on all workers by using the AllReduce
   collective.
3. Update the model with the averaged gradient.

In Bagua, this algorithm is supported via the `GradientAllReduce` algorithm
class. The performance of the `GradientAllReduce` implementation in Bagua by
default should be on par with PyTorch DDP and faster than Horovod in most cases.
Bagua supports additional optimizations such as hierarchical communication that
can be configured when instantiating the `GradientAllReduce` class. They can
make Bagua faster than other implementations in certain scenarios, for example
when the inter-machine network is a bottleneck.
