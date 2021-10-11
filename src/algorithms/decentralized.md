
# Decentralized SGD


## Overview of decentralized training
Decentralized SGD is a data-parallel distributed learning algorithm that removes the requirement of a centralized global model among all workers, which makes it quite different from Allreduce-based or Parameter Server-based algorithms regarding the communication pattern. With decentralized SGD, each worker only needs to exchange data with one or a few specific workers, instead of aggregating data globally. Therefore, decentralized communication has much fewer communication connections than Allreduce, and a more balanced communication overhead than Parameter Server. Although decentralized SGD may lead to different models on each worker, it has been proved in theory that the convergence rate of the decentralized SGD algorithm is the same as its centralized counterpart. You can find the detailed analysis about decentralized SGD in our [paper](https://arxiv.org/abs/1705.09056).


## Decentralized training algorithms

Currently, there are lots of decentralized training algorithms being proposed every now and then. These amazing works are focused on different aspects of decentralized training, like peer selection, data compression, asynchronization and so on, and provide many promising insights. So far Bagua has incorporated two basic decentralized algorithms, i.e., **Decentralized SGD** and [Low Precision Decentralized SGD](./low-precision-decentralized.md). With Bagua's automatic system support for decentralization, we are expecting to see increasingly more decentralized algorithms being implemented in the near future.

## Decentralized SGD

Now we are going to describe the decentralized SGD algorithm implemented in Bagua. Let's assume the number of workers is $n$, the model parameters on worker $i$ is ${\bf x}^{(i)}, i \in \{0,...,n-1\}$. Each worker is able to send or receive data directly from any other workers. In each iteration $t$, the algorithm repeats the following steps:

1. Each worker $i$ calculate the local gradients of iteration $t$: ${\bf g}_t^{(i)}$.
2. Average the local model with its selected peer's model (denote as ${\bf x}_t^{(j)}$), i.e., ${\bf x}_{t+\frac{1}{2}}^{(i)} = \frac{{\bf x}_t^{(i)} + {\bf x}_t^{(j)}}{2}$.
3. Update the averaged model with the local gradients. ${\bf x}_{t+1}^{(i)} = {\bf x}_{t+\frac{1}{2}}^{(i)} - \gamma {\bf g}_t^{(i)}$.

In step 2, we adopt a strategy to select a peer for each worker in each iteration, such that all workers are properly paired and the data exchanging is efficient in the sense that each worker could exchange data with a different peer between iterations. In short, our strategy evenly split workers into two groups, and dynamically pair workers between two groups, varying from iteration to iteration.


## Communication overhead

The communication overhead of decentralized SGD is highly related to the degree of network, i.e., the number of connections a worker has to other workers. Different topologies or strategies will lead to different degrees of the network. It's obvious that the Decentralized SGD algorithm we described before has a network degree of 1. Therefore, in each iteration, a worker only needs to build one connection with one worker to exchange one time of the model size. We compare the communication complexities of different communication patterns regarding the latency and bandwidth of the busiest node.

| Algorithm     | Latency complexity | Bandwidth complexity  |
| :-------------: |:-------------:| :-----:|
| Allreduce (Ring)      | $\mathcal{O}(n)$ | $\mathcal{O}(1)$ |
| Parameter Server      | $\mathcal{O}(1)$ | $\mathcal{O}(n)$ |
| Decentralized SGD in Bagua | $\mathcal{O}(1)$ | $\mathcal{O}(1)$ |

## Benchmark

Given the optimal communication complexity of Decentralized SGD, it can be much faster than its centralized counterparts during the training, especially when the network is slow. We provide some benchmark results [here](../benchmark/index.md) to compare the performance of Decentralized SGD of Bagua with other SOTA systems.


## Example usage

A complete example of running Decentralized SGD can be found at [Bagua examples](https://github.com/BaguaSys/bagua/tree/master/examples/benchmark)
with `--algorithm decentralized` command line argument.

You need to initialize the Bagua algorithm with (see [API documentation](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/algorithms/decentralized/index.html) for what parameters you can customize):

```python
from bagua.torch_api.algorithms import decentralized
algorithm = decentralized.DecentralizedAlgorithm()
```

Then decorate your model with:

```python
model = model.with_bagua([optimizer], algorithm)
```





