# Introduction

Bagua is a distributed training utility developed by [AI Platform@Kuaishou Technology](https://www.kuaishou.com/en) and [DS3 Lab@ETH](https://ds3lab.inf.ethz.ch/). Users can extend the training on a single GPU to multi-GPUs (may across multiple machines), with excellent speedup guarantee, by simply adding a few lines of code. Bagua also provides a flexible system abstraction that supports state-of-the-art system relaxation techniques of distributed training. Powered by the new system design, Bagua has a great ability to implement and extend various state-of-the-art distributed learning algorithms. Researchers can easily develop new distributed training algorithms based on bagua, without sacrificing system performance.

So far, Bagua has integrated primitives including

- Centralized Synchronous Communication (AllReduce)
- Decentralized Synchronous Communication
- Low Precision Communication

Its effectiveness has been verified in various scenarios, including VGG and ResNet on ImageNet, Bert Large and many industrial applications at Kuaishou.

## Performance

<center>
    <img src="./benchmark/figures/scalability_vgg16.png" width="350"/>
    <figcaption>The scalability of different systems on VGG16 with up to 128 GPUs.</figcaption>
</center>

<br/>
<br/>

<center>
    <img src="./benchmark/figures/tradeoff_network_bert-large-bandwidth.png" width="350"/><img src="./benchmark/figures/tradeoff_network_bert-large-latency.png" width="330"/>
    <figcaption>Epoch time of BERT-Large Finetune under different network conditions for different systems.</figcaption>
</center>

For more comprehensive and up to date results, refer to [Bagua benchmark page](https://baguasys.github.io/tutorials/benchmark/index.html).

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
* [Bagua Examples](https://github.com/BaguaSys/examples)
* [Bagua Tutorials](https://bagua-tutorials.kwai-seattle.com/)
* [Bagua API Documentation](https://bagua.readthedocs.io/)
