# Frequently Asked Questions and Troubleshooting

**Q1: NCCL_ERROR_INVALID_ARGUMENT.**

A1: This is caused by the NCCL version conflict，update your torch==1.7.0 and torchvision==0.8.0.

**Q2: Error when installing rust: error: could not rename component file from '/root/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/share/doc/cargo' to '/root/.rustup/tmp/m74fkrv0gv6708f6_dir/bk'
error: caused by: other os error.**

A2: Clean the original installation record first by `rm -rf /root/.rustup` and reinstall.

**Q3: Out of memory when using the Quantize algorithm.**

A3: The Quantize algorithm compresses the communication content and requires a certain amount of additional memory, so reduce the batch size appropriately.

**Q4: Got stuck when running a multi-machine multi-card program.**

A4: You can try to check whether the machine has multiple network cards, and use command `NCCL_SOCKET_IFNAME=network card name (such as eth01)` to specify the network card. Multi-network card information can be judged by command `ls /sys/class/net/`.

**Q5: After using bagua, no matter which algorithm runs, Loss almost never drops.**

A5: Check whether horovod is imported in the training script at the same time, and comment the `import horovod` out.

**Q6: After increasing batch size by multi-machine multi-card training, model perfomance decreases.**

Tricks you can try:
    
1. Train more epochs and increase the number of training iterations to 1.2-1.3 times than the original.
    
2. Linearly adjust learning rate (lr). If the total batch_size of distributed training is increased by N times compared with the original, the initial lr of multi-machine multi-card training should also increased by N times and becomes N*lr.
    
3. Gradual warmup. Perform a Gradual warmup for several epochs before the offical training starts. The learning rate at this stage gradually increases. The initial learning rate is set to lr which used in the original training process, the target learning rate is set to N*lr, and the warmup stage increases linearly with the training step. . (Ps: You can also try a constant warmup with a fixed lr, but the effect should not be as good as the Gradual warmup according to [Accurate, Large Minibatch SGD:
Training ImageNet in 1 Hour](https://arxiv.org/pdf/1706.02677.pdf). Usually 2 and 3 need to be used together, which more likely to achieve a better result.)
    
    
**Q7: The loss drops slowly when using the decentralize algorithm.**

A7: The learning rate can be expanded by 10 times compared to allreduce (horovod). Of course, the expansion factor can be appropriately adjusted (1-10) according to the number of GPU cards used.

**Q8: When using decentralize, the loss of the first few epochs drops, and then it hardly drops.**

A8: When the loss is flat, learning rate can be further reduced (for example, using torch's lr_scheduler), which is the same for any optimization algorithm.
