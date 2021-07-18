
# Decentralized SGD


## Overview of decentralized training
Decentralized SGD is a data-parallel distributed learning algorithm that removes the requirement of a centralized global model among all workers, which makes it quite different from Allreduce-based or Parameter Server-based algorithms regarding the communication pattern. With decentralized SGD, each worker only needs to exchange data with one or a few specific workers, instead of aggregating data globally. Therefore, decentralized communication has much fewer communication connections than Allreduce, and a more balanced communication overhead than Parameter Server. Although decentralized SGD may lead to different models on each worker, it has been proved in theory that the convergence rate of the decentralized SGD algorithm is the same as its centralized counterpart. You can find the detailed analysis about decentralized SGD in our paper [[1]](https://arxiv.org/abs/1705.09056)[[2]](https://arxiv.org/abs/1803.06443).


## Decentralized training algorithms

Currently, there are lots of decentralized training algorithms being proposed every now and then. These amazing works are focused on different aspects of decentralized training, like peer selection, data compression, asynchronization and so on, and provide many promising insights. So far Bagua has incorporated two basic decentralized algorithms, i.e., the full precision **Decentralized SGD** and the low precision **Difference Compression Decentralized SGD**. With Bagua's automatic system support for decentralization, we are expecting to see increasingly more decentralized algorithms being implemented in the near future.

## Decentralized SGD

Now we are going to describe the decentralized SGD algorithm implemented in Bagua. Let's assume the number of workers is $n$, the model parameters on worker $i$ is $\bf x^{(i)}, i \in \{0,...,n-1\}$. Each worker is able to send or receive data directly from any other workers. In each iteration $t$, the algorithm repeats the following steps:

1. Each worker $i$ calculate the local gradients of iteration $t$: $\bf g_t^{(i)}$.
2. Average the local model with its selected peer's model (denote as $\bf x_t^{(j)}$), i.e., $\bf x_{t+\frac{1}{2}}^{(i)} = \frac{\bf x_t^{(i)} + \bf x_t^{(j)}}{2}$.
3. Update the averaged model with the local gradients. $\bf x_{t+1}^{(i)} = \bf x_{t+\frac{1}{2}}^{(i)} - \gamma \bf g_t^{(i)}$.

In step 2, we adopt a strategy to select a peer for each worker in each iteration, such that all workers are properly paired and the data exchanging is efficient in the sense that each worker could exchange data with a different peer between iterations. In short, our strategy evenly split workers into two groups, and dynamically pair workers between two groups, varying from iteration to iteration.

## Difference Compression Decentralized SGD

The difference compression decentralized SGD follows the same assumptions with decentralized SGD. Besides, each worker stores model replicas of its connected peers $\left \{ {\bf \hat x^{(j)}}: j \text{ is worker }i \text{'s peer} \right \}$. At each iteration $t$, the algorithm repeats the following steps:

1. Each worker $i$ calculate the local gradients of iteration $t$: $\bf g_t^{(i)}$.
2. Update the local model using local stochastic gradient and the weighted average of its connected peers' replicas, ${\bf x_{t+\frac{1}{2}}^{(i)}} = \sum_{j=1}^{n} W_{ij} {\bf x_{t+1}^{(i)}} - \gamma {\bf g_t^{(i)}}$.
3. Compute the difference ${\bf z_{t}^{(i)} = x_{t+\frac{1}{2}}^{(i)} - x_{t}^{(i)}}$, and quantize it into $Q( {\bf z_{t}^{(i)}})$ with a quantization function $Q( \cdot )$.
4. Update the local model,  ${\bf x_{t+1}^{(i)} = x_{t}^{(i)} + Q(z_{t}^{(i)})}$.
5. Send $Q ( {\bf z_{t}^{(i)}} )$ to its connected peers, and update its connected peers' replicas, ${\bf \hat x_{t+1}^{(j)} =\hat x_{t}^{(j)} + Q(z_{t}^{(j)}) }$.


Since each worker need to store the model replicas of its connected peers, once the peers of a worker is determined, they should not be changed during the whole process.

## Communication overhead

The communication overhead of decentralized SGD is highly related to the degree of network, i.e., the number of connections a worker has to other workers. Different topologies or strategies will lead to different degrees of the network. It's obvious that the Decentralized SGD algorithms we described before have a network degree of $k$, where $k$ is the number of peers for each worker. Therefore, in each iteration, a worker only needs to build $k$ connections with its peers, exchanging one time of the model size each. We compare the communication complexities of different communication patterns regarding the latency and bandwidth of the busiest node.

| Algorithm     | Latency complexity | Bandwidth complexity  |
| :-------------: |:-------------:| :-----:|
| Allreduce (Ring)      | $\mathcal{O}(n)$ | $\mathcal{O}(1)$ |
| Parameter Server      | $\mathcal{O}(1)$ | $\mathcal{O}(n)$ |
| Decentralized Algorithms in Bagua | $\mathcal{O}(k)$ | $\mathcal{O}(k)$ |

Since $k$ is much smaller than $n$, we can show that decentralized algorithms can greatly outperform its centralized counterparts.

## Benchmark

Given the optimal communication complexity of Decentralized SGD, it can be much faster than its centralized counterparts during the training, especially when the network is slow. We provide some benchmark results [here](../benchmark/index.md) to compare the performance of Decentralized SGD of Bagua with other SOTA systems.


## Example

### Decentralized SGD
A complete example of running Decentralized SGD can be found at [Bagua examples](https://github.com/BaguaSys/examples/blob/main/benchmark/synthetic_benchmark.py)
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
### Difference Compression Decentralize SGD
To use Difference Compression Decentralize SGD, initialize the Bagua algorithm with:

```python
from bagua.torch_api.algorithms import decentralized
algorithm = decentralized.LowPrecisionDecentralizedAlgorithm()  
```

Others are the same with the example using Decentralized SGD.

More examples can be found at [Bagua examples](https://github.com/BaguaSys/examples) with  `--algorithm low_precision_decentralized`.






