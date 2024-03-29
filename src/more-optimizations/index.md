# System Optimizations

Besides distributed communication algorithms, Bagua supports many features to
further accelerate your training workload.

- [**Cached Dataset**](cached-dataset.md): When samples in a dataset need tedious preprocessing, or reading the dataset itself is slow, they could become a major bottleneck of the whole training process. Bagua provides cached dataset to speedup this process by caching data samples in memory, so that reading these samples after the first time can be much faster.
- [**TCP Communication Acceleration (Bagua-Net)**](./bagua-net.md): Bagua-Net is a low level communication acceleration feature provided by Bagua. It can greatly improve the throughput of AllReduce on TCP network. You can enable Bagua-Net optimization on any distributed training job that uses NCCL to do GPU communication (this includes PyTorch-DDP, Horovod, DeepSpeed, and more).
- [**Performance Autotuning**](../performance-autotuning/index.md): Bagua can automatically tune system parameters to achieve the highest throughput.
- [**Generic Fused Optimizer**](generic-fused-optimizer.md): Bagua provides generic fused optimizer which improve the performance of optimizers by fusing the optimizer `.step()` operation on multiple layers. It can be applied to arbitrary PyTorch optimizer, in contrast to [NVIDIA Apex](https://nvidia.github.io/apex/optimizers.html)'s approach, where only some specific optimizers are implemented.
- [**Load Balanced Data Loader**](load-balanced-data-loader.md): When the computation complexity of samples in training data are different, for example in NLP and speech tasks, where each sample have different lengths, distributed training throughput can be greatly improved by using Bagua's load balanced data loader, which distributes samples in a way that each worker's workload are similar.

