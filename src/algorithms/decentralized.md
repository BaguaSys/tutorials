
# Decentralized SGD


## Overview
Decentralized SGD is a data-parallel distributed learning algorithm that removes the requirement of a centralized global model among all workers, which makes it quite different from Allreduce-based or Parameter Server-based algorithms regarding the communication pattern. With decentralized SGD, each worker only needs to exchange data with one or a few specific workers, instead of aggregating data globally. Therefore, decentralized communication has much less communication connections than Allreduce, and more balanced communication overhead than Parameter Server. Although decentralized SGD may lead to different models on each worker, it has been proved in theory that the convergence rate of the decentralized SGD is same as its centralized counterpart. You can find the detailed analysis about decentralized SGD in our [paper](https://arxiv.org/abs/1705.09056).


## Decentralized Training Algorithms

Currently, there are lots of decentralized training algorithms being proposed every now and then. These amazing works are focused on different aspects of the decentralized training, like peer selection, data compression, asynchronization and so on, and provide many promising insights. So far Bagua has already incorporated two decentralized algorithms: 1) Decentralized SGD and 2) Decentralized SGD with compression. With Bagua's automatic system support to decentralization, we are expecting to see increasingly more decentralized algorithms to be implemented in the near future.

The following of this tutorial will focus on the  Decentralized SGD algorithm. For Decentralized SGD with compression, please refer to another tutorial [here](./decen_compressed.md).

## Decentralized SGD with 

Now we are going to describe the decentralized SGD algorithm implemented in Bagua. Assume the number of workers is $n$, the model parameters on worker $i$ is $\bf x_i$


Each worker is able to send or receive data from any other works. The algorithm can be summarized in the following steps:

1. Each worker calculate the local gradients $\bf g_i$


## Communication Overhead

The communication overhead of decentralized SGD is highly related to the degree of network, i.e., the number of connections a worker has to other workers. Different topologies will lead to different degrees of network. For example, in the ring topology, each worker has two connections with its two neighbors; in the random-probing topology, each worker only needs one connection with another random worker. 

Let's assume there are $n$ workers in total. 









## Example




