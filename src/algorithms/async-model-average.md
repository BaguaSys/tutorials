# Asynchronous Model Average

Let's talk about our first asynchronous algorithm. In the algorithms discussed above, communications are performed
synchronously at certain stage during the training process. In [Gradient AllReduce](./gradient-allreduce.md) and
[ByteGrad](./bytegrad.md) algorithm, gradients (or quantized gradients) are exchanged among workers at the end of
backward process of each iteration. [QAdam](./q-adam.md) updates its momentum locally, then exchanges them at similar
stage before applying them to its optimizer. [Decentralized](./decentralized.md) algorithm plays a bit different,
it initiates model averaging at the beginning of each forward process and updates averaged weights at the end of backward.

These synchronous algorithms we mentioned here all require the worker itself to identify which iteration and what
stage it lies in, so as to perform communications simultaneously. Asynchronous algorithm does not impose such
restrictions. Communications are done periodically by a looping thread in parallel with model computations. The number
of iterations on each worker might be different. Faster workers run more and slower workers run less.

Asynchronous algorithm is especially useful in heterogeneous environment, where stragglers are known to slow down the
whole training process for synchronous algorithms.

## Algorithm

The asynchronous algorithm can be described in the following: each worker maintains a local model ${\bf x}$ in local memory
and repeats the following steps:

1. **Compute gradients**: Each worker $i$ calculates its local gradients $\nabla F(\hat {\bf x}^{(i)})$, where
$\hat {\bf x}^{(i)}$ is read from the model in the local memory.
2. **Gradient update**: Update the model in the local memory by
${\bf x}^{(i)} = {\bf x}^{(i)} - \gamma \nabla F(\hat {\bf x}^{(i)}) $, note that $\hat {\bf x}^{(i)}$ may not be the
same with ${\bf x}^{(i)}$ as it can be modified in the **averaging** step.
3. **Averaging**: Average local model ${\bf x}^{(i)}$ with all other workers' models. More specifically,
 $ {\bf x}^{(i)} = \frac{1}{n} \sum_{i'=1}^{n} {\bf x}^{(i')} $.

All workers run the procedure above simultaneously.

**Gradient update** and **Averaging** can run in parallel and the averaging step needs to be atomic.

## Example usage

A complete example of running Asynchronous Model Average can be found at [Bagua examples](https://github.com/BaguaSys/examples/blob/main/benchmark/synthetic_benchmark.py)
with `--algorithm async` command line argument.

You need to initialize the Bagua algorithm with (see [API documentation](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/algorithms/async_model_average/index.html) for further customization):

```python
from bagua.torch_api.algorithms import async_model_average
algorithm = async_model_average.AsyncModelAverageAlgorithm()
```

Then decorate your model with:

```python
model = model.with_bagua([optimizer], algorithm)
```

Unlike running synchronous algorithms, you need to stop asynchronous communications explicitly at the end of your training by:

```python
algorithm.abort(model)
```
or
```python
model.bagua_algorithm.abort(model)
```