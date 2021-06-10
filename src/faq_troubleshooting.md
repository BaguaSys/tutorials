# Frequently Asked Questions and Troubleshooting

## NCCL_ERROR_INVALID_ARGUMENT

Usually this is caused by the NCCL version conflict, update your `torch>=1.7.0`, and `torchvision>=0.8.0`.

## Error when installing rust

If you see some error like the message below, just clean the original installation record first by `rm -rf /root/.rustup` and reinstall.

```shell
$ error: could not rename component file from '/root/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/share/doc/cargo' to '/root/.rustup/tmp/m74fkrv0gv6708f6_dir/bk'error: caused by: other os error.
```

## Out of memory when using the quantize algorithm

The quantize algorithm compresses the communication content and requires a certain amount of additional memory, so reduce the batch size appropriately.

## Got stuck when running a distributed program

You can try to check whether the machine has multiple network cards, and use command `NCCL_SOCKET_IFNAME=network card name (such as eth01)` to specify the network card. Multi-network card information can be judged by command `ls /sys/class/net/`.

## After using bagua, no matter which algorithm runs, loss almost never drops

Check whether horovod is imported in the training script at the same time, and comment the `import horovod` out.

## After increasing batch size by distributed training, model perfomance decreases

We provid some tricks you can try:
1. Train more epochs and increase the number of training iterations to <u>1.2-1.3 times</u> than the original.
   
2. <u>Linearly adjust learning rate</u>. If the total batch size of distributed training is increased by N times compared with the original, the learning rate (lr) of multi-machine multi-card training should also increased by N times and becomes <u>N*lr</u>.
   
3. <u>Gradual warmup</u>. Perform a Gradual warmup for several epochs before the offical training starts. The initial learning rate is set to lr which used in the original training process, and the target learning rate is set to N*lr. The learning rate at this stage  <u>increases linearly with the training step</u>. After the warmup, you can go back to the original learning rate schedule. (Ps: You can also try a constant warmup with a fixed lr, but the effect should not be as good as the gradual warmup according to [Accurate, Large Minibatch SGD:
Training ImageNet in 1 Hour](https://arxiv.org/pdf/1706.02677.pdf). <u>Usually 2 and 3 need to be used together</u>, which more likely to achieve a ideal result.)

## The loss drops slowly when using the decentralize algorithm

The learning rate can be expanded by 10 times compared to allreduce (horovod). Of course, the expansion factor can be appropriately adjusted (1-10) according to the number of GPU cards used.

## When using decentralize, the loss of the first few epochs drops, and then it hardly drops

When the loss is flat, learning rate can be further reduced (for example, using torch's lr_scheduler), which is the same with other optimization algorithm.

