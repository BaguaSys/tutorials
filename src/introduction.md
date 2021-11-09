<p align="center">
<img src="https://user-images.githubusercontent.com/18649508/137567125-e949ae92-8c72-41ba-9a8e-6108e33005fc.png"/>
</p>

Bagua is a deep learning training acceleration framework for PyTorch developed by [AI platform@Kuaishou Technology](https://www.kuaishou.com/en) and [DS3 Lab@ETH](https://ds3lab.inf.ethz.ch/). Bagua currently supports:

- **Advanced Distributed Training Algorithms**: Users can extend the training on a single GPU to multi-GPUs (may across multiple machines) by simply adding a few lines of code (optionally in [elastic mode](https://tutorials.baguasys.com/elastic-training/)). One prominent feature of Bagua is to provide a flexible system abstraction that supports state-of-the-art system relaxation techniques of distributed training. So far, Bagua has integrated communication primitives including
  - Centralized Synchronous Communication (e.g. [Gradient AllReduce](./algorithms/gradient-allreduce.md))
  - Decentralized Synchronous Communication (e.g. [Decentralized SGD](./algorithms/decentralized.md))
  - Low Precision Communication (e.g. [ByteGrad](./algorithms/bytegrad.md))
  - Asynchronous Communication (e.g. [Async Model Average](./algorithms/async-model-average.md))
- [**Cached Dataset**](./more-optimizations/cached-dataset.md): When samples in a dataset need tedious preprocessing, or reading the dataset itself is slow, they could become a major bottleneck of the whole training process. Bagua provides cached dataset to speedup this process by caching data samples in memory, so that reading these samples after the first time can be much faster.
- [**TCP Communication Acceleration (Bagua-Net)**](./more-optimizations/bagua-net.md): Bagua-Net is a low level communication acceleration feature provided by Bagua. It can greatly improve the throughput of AllReduce on TCP network. You can enable Bagua-Net optimization on any distributed training job that uses NCCL to do GPU communication (this includes PyTorch-DDP, Horovod, DeepSpeed, and more).
- [**Performance Autotuning**](./performance-autotuning/index.md): Bagua can automatically tune system parameters to achieve the highest throughput.
- [**Generic Fused Optimizer**](./more-optimizations/generic-fused-optimizer.md): Bagua provides generic fused optimizer which improve the performance of optimizers by fusing the optimizer `.step()` operation on multiple layers. It can be applied to arbitrary PyTorch optimizer, in contrast to [NVIDIA Apex](https://nvidia.github.io/apex/optimizers.html)'s approach, where only some specific optimizers are implemented.
- [**Load Balanced Data Loader**](./more-optimizations/load-balanced-data-loader.md): When the computation complexity of samples in training data are different, for example in NLP and speech tasks, where each sample have different lengths, distributed training throughput can be greatly improved by using Bagua's load balanced data loader, which distributes samples in a way that each worker's workload are similar.

Its effectiveness has been validated in various scenarios and models, including VGG and ResNet on ImageNet, Bert Large, and multiple huge scale industrial applications at Kuaishou such as 

- the recommendation system supporting model training with dozens of TB parameters,
- video/image understanding with >1 billion images/videos,
- ASR with TB level datasets,
- etc.

## Performance

<!-- <center>
    <img src="./benchmark/figures/scalability_vgg16.png" width="350"/>
    <figcaption>The scalability of different systems on VGG16 with up to 128 GPUs.</figcaption>
</center>

<br/>
<br/> -->

<center>
    <img src="./benchmark/figures/e2e_vgg16_128.png" width="600"/> 
    <figcaption>The performance of different systems and algorithms in Bagua on VGG16 with 128 GPUs and different network bandwidth.</figcaption> 
</center>

<br/>
<br/>

<center>
    <img src="./benchmark/figures/tradeoff_network_bert-large-bandwidth.png" width="350"/><img src="./benchmark/figures/tradeoff_network_bert-large-latency.png" width="330"/>
    <figcaption>Epoch time of BERT-Large Finetune under different network conditions for different systems.</figcaption>
</center>

For more comprehensive and up to date results, refer to [Bagua benchmark page](benchmark/index.html).

## Cite Bagua

```bibtex
@misc{gan2021bagua,
      title={BAGUA: Scaling up Distributed Learning with System Relaxations}, 
      author={Shaoduo Gan and Xiangru Lian and Rui Wang and Jianbin Chang and Chengjun Liu and Hongmei Shi and Shengzhuo Zhang and Xianghong Li and Tengxu Sun and Jiawei Jiang and Binhang Yuan and Sen Yang and Ji Liu and Ce Zhang},
      year={2021},
      eprint={2107.01499},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Links

* [Bagua Github](https://github.com/BaguaSys/bagua)
* [Bagua Examples](https://github.com/BaguaSys/bagua/tree/master/examples)
* [Bagua Tutorials](https://tutorials.baguasys.com/)
* [Bagua API Documentation](https://bagua.readthedocs.io/)
