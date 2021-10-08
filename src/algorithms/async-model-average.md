# Asynchronous Model Average

In synchronous communication algorithms such as [Gradient AllReduce](./gradient-allreduce.md),
every worker needs to be in the same iteration in a lock-step style. When there is no straggler
in the system, such synchronous algorithms are reasonably efficient, and gives deterministic training results
that are easier to reason about. However, when there are stragglers in the system, with synchronous algorithms
faster workers have to wait for the slowest worker in each iteration, which can dramatically harm
the performance of the whole system. To deal with stragglers, we can use asynchronous 
algorithms where workers are not required to be synchronized. The *Asynchronous Model Average* algorithm provided by Bagua is one of such algorithms.

## Algorithm

The *Asynchronous Model Average* algorithm can be described as follows: 

Every worker maintains a local model ${\bf x}$. The $i$-th worker maintains ${\bf x}^{(i)}$. Every worker runs two threads in parallel. The first thread does gradient computation (called computation thread) and the other one does communication (called communication thread). For each worker $i$, a lock $m_i$ controls the access to its model.

The computation thread on the $i$-th worker repeats the following steps:

1. Acquire lock $m_i$.
2. Calculate a local gradient $\nabla F({\bf x}^{(i)})$ on a batch of input data.
3. Release lock $m_i$.
4. Update the model with local gradient,
${\bf x}^{(i)} = {\bf x}^{(i)} - \gamma \nabla F({\bf x}^{(i)})$.

The communication thread on the $i$-th worker repeats the following steps:

1. Acquire lock $m_i$.
2. Average local model ${\bf x}^{(i)}$ with all other workers' models: ${\bf x}^{(i)} =  \frac{1}{n} \sum_{j=1}^{n} {\bf x}^{(j)}$.
3. Release lock $m_i$.

Every worker run the two threads independently and concurrently.

## Example usage

First initialize the Bagua algorithm (see [API documentation](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/algorithms/async_model_average/index.html) for more options):

```python
from bagua.torch_api.algorithms import async_model_average
algorithm = async_model_average.AsyncModelAverageAlgorithm()
```

Then use the algorithm for the model

```python
model = model.with_bagua([optimizer], algorithm)
```

Unlike running synchronous algorithms, you need to stop the communication thread explicitly when the training process is done (for example when you want to run test):

```python
model.bagua_algorithm.abort(model)
```

To resume the communication thread when you start training again, do:
```python
model.bagua_algorithm.resume(model)
```

A complete example of running the *Asynchronous Model Average* algorithm can be found in [Bagua examples](https://github.com/BaguaSys/examples/blob/main/benchmark/synthetic_benchmark.py)
with `--algorithm async` command line argument.

