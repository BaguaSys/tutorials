# Frequently Asked Questions and Troubleshooting

<!-- ## `NCCL_ERROR_INVALID_ARGUMENT` -->

<!-- Usually this is caused by the library version conflict, try updating to `torch>=1.7.0`. -->

## Dataloader sometimes hang when `num_workers` > 0

Add `torch.multiprocessing.set_start_method("forkserver")`. The default `"fork"` strategy is error prone by design. For more information, see [PyTorch documentation](https://pytorch.org/docs/stable/notes/multiprocessing.html#avoiding-and-fighting-deadlocks), and [StackOverflow](https://stackoverflow.com/questions/64095876/multiprocessing-fork-vs-spawn).

## Error when installing Rust

If you see some error like the message below, just clean the original installation record first by `rm -rf /root/.rustup` and reinstall.

```shell
error: could not rename component file from '/root/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/share/doc/cargo' to '/root/.rustup/tmp/m74fkrv0gv6708f6_dir/bk'error: caused by: other os error.
```

## Error when compiling bagua_core

If some of the error messages below are issued during your compiling bagua_core, make sure `CUDA_LIBRARY_PATH` environment variable is correctly set, and the CUDA library path is added to `LD_LIBRARY_PATH` environment variable as well. These environment variables should be correctly set before you installing MPI and hwloc.

```
error: empty search path given via `-L`
error: could not compile `bagua-core-internal`
```
<!-- ## Out of memory when using the quantize algorithm -->

<!-- The quantize algorithm compresses the communication content and requires a certain amount of additional memory. Reduce the batch size appropriately. -->

## Hang when running a distributed program

You can try to check whether the machine has multiple network interfaces, and
use command `NCCL_SOCKET_IFNAME=network card name (such as eth01)` to specify
the one you want to use (usually a physical one). Card information can be
obtained by `ls /sys/class/net/`.

Setting `NCCL_DEBUG` to “WARN” will make NCCL print an explicit warning message 
before returning the error, which will help you idenfity the error.

## Model accuracy drops

Using a different algorithm or using more GPUs has similar effect as using a
different optimizer, so you need to retune your hyperparameters. Some tricks you
can try:
1. Train more epochs and increase the number of training iterations to
   0.2-0.3 times more than the original.
2. Scale the learning rate. If the total batch size of distributed training is
   increased by $N$ times, the learning rate should also be increased by $N$
   times to be $N \times lr$.
3. Performing a gradual learning rate warmup for several epochs often helps (see
   also [Accurate, Large Minibatch SGD: Training ImageNet in 1
   Hour](https://arxiv.org/pdf/1706.02677.pdf)).

<!-- ## The loss drops slowly when using the decentralized algorithm -->

<!-- Decentralized algorithms often need a larger learning rate compared to other algorithms. -->
