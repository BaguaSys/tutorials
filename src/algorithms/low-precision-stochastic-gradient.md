# Low Precision Stochastic Gradient

Large scale distributed training requires significant communication cost, which
is especially true for large models. For example in traditional synchronous
distributed setup with AllReduce to synchronize gradients (which is the case for
many libraries, such as [Horovod](https://github.com/horovod/horovod) and
[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)), in
each iteration of the training process, the gradient, whose size is equal to the
model size, needs to be sent and received on each worker. Such communication
cost soon becomes the training bottleneck in many scenarios. 

Bagua provides a built-in gradient compression algorithm, which compresses the
gradient floats to 8bit bytes before communication, which saves 3/4 of the
original cost.
