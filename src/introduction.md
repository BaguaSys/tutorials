# Introduction

Bagua is a distributed training utility developed by [Kuaishou Technology](https://www.kuaishou.com/en) and [DS3 Lab@ETH](https://ds3lab.inf.ethz.ch/). Users can extend the training on a single GPU to multi-GPUs (may across multiple machines), with excellent speedup guarantee, by simply adding a few lines of code. Bagua also provides a flexible system abstraction that supports state-of-the-art system relaxation techniques of distributed training. Powered by the new system design, Bagua has a great ability to implement and extend various state-of-the-art distributed learning algorithms. Researchers can easily develop new distributed training algorithms based on bagua, without sacrificing system performance.

So far, Bagua has integrated primitives including

- Centralized Synchronous Communication (AllReduce)
- Decentralized Synchronous Communication
- Low Precision Communication

Its effectiveness has been verified in various scenarios, including VGG and ResNet on ImageNet, Bert Large and many industrial applications at Kuaishou.

# Links

* [Bagua Github](https://github.com/BaguaSys/bagua)
* [Bagua Tutorials](https://baguasys.github.io/tutorials)
* [Bagua API Documentation](https://bagua.readthedocs.io/)
