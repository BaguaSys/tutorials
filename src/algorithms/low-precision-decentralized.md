# Low Precision Decentralized SGD

## Overview

As the name suggests, low precision decentralized SGD combines decentralized training and quantized training together. It follows the framework of [decentralized SGD](./decentralized.md) that each worker does not need to aggregate data globally, but to exchange data with few workers, more specifically, its peers. Thus this algorithm has similar communication overhead with decentralized SGD. The latency complexity and bandwidth complexity of low precision decentralized SGD are both $O(k)$, where $k$ is the number of peers. This is consistent with our analysis in decentralized SGD, where we consider a special case that each worker has only one peer.

With communication compression, low precision decentralized SGD can reduce communication overhead further. It should be noted that data exchanged between workers are not compressed local models, but the compressed differences of local models between two successive iterations. In this way, the low precision decentralized SGD algorithm can achieve the same convergence rate with decentralized SGD, as well as full precision centralized ones. Detailed proof can be found in [this paper](https://arxiv.org/abs/1803.06443).

Benefiting from both decentralization and communication compression, low precision decentralized SGD is particular useful in high communication latency and low network bandwidth scenarios.

## Algorithm

Assume the number of workers is $n$, and the model parameters on worker $i$ is $\bf x^{(i)}$, $i \in \left \{0,...,n-1 \right \}$. Each worker stores model replicas of its connected peers $\left \{ {\bf \hat x^{(j)}}: j \text{ is worker }i \text{'s peer} \right \}$ and is able to send data to or receive data from its peers. At each iteration $t$, the algorithm repeats as follows:

1. Calculate the gradient on worker $i$: $\bf g_t^{(i)}$.
2. Update the local model using local stochastic gradient and the weighted average of its connected peers' replicas:
         ${\bf x_{t+\frac{1}{2}}^{(i)}} = \sum_{j=1}^{n} W_{ij} {\bf x_{t+1}^{(i)}} - \gamma {\bf g_t^{(i)}}$.
3. Compute the difference ${\bf z_{t}^{(i)} = x_{t+\frac{1}{2}}^{(i)} - x_{t}^{(i)}}$, and quantize it into $Q( {\bf z_{t}^{(i)}})$ with a quantization function $Q( \cdot )$.
4. Update the local model with compressed difference,  ${{\bf x_{t+1}^{(i)}} ={\bf x_{t}^{(i)}} + Q({\bf z_{t}^{(i)}})}$.
5. Send $Q ( {\bf z_{t}^{(i)}} )$ to its connected peers, and update its connected peers' replicas with compressed differences it received, ${{\bf \hat x_{t+1}^{(j)}} ={\bf \hat x_{t}^{(j)}} + Q({\bf z_{t}^{(j)}}) }$.

The quantization function $Q(\cdot)$ calculates the minimum value $x$ and maximum value $y$ of its input, and the split $[x, y]$ into evenly spaced 256 intervals. Then represent each element of its input by a 8bit integer representing which interval the original element is in.

Each worker stores model replicas of its connected peers, once the peers of a worker is determined, they should not be changed during the whole process.

## Example usage

A complete example of running Decentralized SGD can be found at [Bagua examples](https://github.com/BaguaSys/examples/blob/main/benchmark/synthetic_benchmark.py)
with `--algorithm low_precision_decentralized` command line argument.

You need to initialize the Bagua algorithm with (see [API documentation](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/algorithms/decentralized/index.html) for further customization):

```python
from bagua.torch_api.algorithms import decentralized
algorithm = decentralized.LowPrecisionDecentralizedAlgorithm()
```

Then decorate your model with:

```python
model = model.with_bagua([optimizer], algorithm)
```
