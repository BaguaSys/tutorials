# Frequently Asked Questions and Troubleshooting

## `NCCL_ERROR_INVALID_ARGUMENT`

Usually this is caused by the library version conflict, try updating to `torch>=1.7.0`.

## Error when installing rust

If you see some error like the message below, just clean the original installation record first by `rm -rf /root/.rustup` and reinstall.

```shell
$ error: could not rename component file from '/root/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/share/doc/cargo' to '/root/.rustup/tmp/m74fkrv0gv6708f6_dir/bk'error: caused by: other os error.
```

## Out of memory when using the quantize algorithm

The quantize algorithm compresses the communication content and requires a certain amount of additional memory. Reduce the batch size appropriately.

## Hang when running a distributed program

You can try to check whether the machine has multiple network cards, and use command `NCCL_SOCKET_IFNAME=network card name (such as eth01)` to specify the network card. Card information can be determined by command `ls /sys/class/net/`.

## After increasing batch size by distributed training, model perfomance decreases

We provid some tricks you can try:
1. Train more epochs and increase the number of training iterations to 0.2-0.3 times more than the original.
2. Scale the learning rate. If the total batch size of distributed training is increased by N times compared with the original, the learning rate (lr) of multi-machine multi-card training should also increased by N times and becomes N*lr.
3. Learning rate warmup. Perform a Gradual learning rate warmup for several epochs (see also [Accurate, Large Minibatch SGD:
Training ImageNet in 1 Hour](https://arxiv.org/pdf/1706.02677.pdf).)

## The loss drops slowly when using the decentralized algorithm

Decentralized algorithms often need a larger learning rate compared to other algorithms.
