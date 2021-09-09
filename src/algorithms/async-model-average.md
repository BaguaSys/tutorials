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

Every worker maintains a local model ${\bf x}$. The $i$-th worker maintains ${\bf x}^{(i)}$. Every worker repeats the following three steps, for the $i$-th worker:

1. **Compute gradients**: Calculate a local gradient $\nabla F({\bf \hat x}^{(i)})$, where $\hat {\bf x}^{(i)}$ is a clone of $\mathbf{x}_i$.
2. **Averaging**: Average local model ${\bf x}^{(i)}$ with all other workers' models by
${\bf x}^{(i)} =  \frac{1}{n} \sum_{j=1}^{n} {\bf x}^{(j)}$.
3. **Gradient update**: Update the model with local gradient,
${\bf x}^{(i)} = {\bf x}^{(i)} - \gamma \nabla F(\hat {\bf x}^{(i)})$.

All workers run the procedure above simultaneously.

**Gradient update** and **averaging** can run in parallel and the **averaging** step needs to be atomic.


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

