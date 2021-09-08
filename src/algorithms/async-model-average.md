# Asynchronous Model Average

In synchronous communication algorithms such as [Gradient AllReduce](./gradient-allreduce.md),
every worker needs to be in the same iteration in a lock-step style. When there is no straggler
in the system, such synchronous algorithms are fast, and often gives deterministic training results
that are easier to reason about. However, when there are stragglers in the system, synchronous algorithms
require faster workers to wait for the slowest one in each iteration, which can dramatically reduce
the overall performance of the whole system. To solve this problem, we can use an asynchronous communication 
algorithm that does not require every worker to be in the same iteration.

The *Asynchronous Model Average* algorithm provided by Bagua is one of such algorithms.

## Algorithm

The *Asynchronous Model Average* algorithm can be described as follows: 

Every worker maintains a local model ${\bf x}$. The $i$-th worker maintains $\hat {\bf x}^{(i)}$.
Every worker runs two threads in parallel, one for computation and the other for communication.
The two threads shares a buffer $w$ initialized by all $0$s.

The computation thread on the $i$-th worker repeats the following three steps:

1. Calculate the local gradient $\nabla F({\bf x}^{(i)})$.
2. Update the model with buffer $w$, ${\bf x}^{(i)} = {\bf x}^{(i)} + w$.
3. Update the model using local gradient, ${\bf x}^{(i)} = {\bf x}^{(i)} - \gamma \nabla F({\bf x}^{(i)}) $.

The communication thread repeats as follows:
1. Average local model ${\bf x}^{(i)}$ with all other workers' models,
 ${\bf \bar x}^{(i)} = \frac{1}{n} \sum_{i'=1}^{n} {\bf x}^{(i')}$.
2. Update shared buffer $w$, $w = w + {\bf \bar x} - {\bf x}^{(i)}$.

## Example usage

First initialize the Bagua algorithm (see [API documentation](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/algorithms/async_model_average/index.html) for more options):

```python
from bagua.torch_api.algorithms import async_model_average
algorithm = async_model_average.AsyncModelAverageAlgorithm()
```

Then use the algorithm for our model

```python
model = model.with_bagua([optimizer], algorithm)
```

Unlike running synchronous algorithms, you need to stop asynchronous communications explicitly when the training process is done with:

```python
algorithm.abort(model)
```
or
```python
model.bagua_algorithm.abort(model)
```

A complete example of running the *Asynchronous Model Average* algorithm can be found in [Bagua examples](https://github.com/BaguaSys/examples/blob/main/benchmark/synthetic_benchmark.py)
with `--algorithm async` command line argument.

