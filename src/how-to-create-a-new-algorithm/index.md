# How to Create a New Algorithm

Thanks to the innovative design of Bagua, algorithm developers now can easily create, test and benchmark their distributed learning algorithms in a realistic system. Within Bagua, developers have the freedom to manipulate almost all the details regarding the data-parallel distributed training, including What to communicate, When to communicate, How to update the model and so on. Besides, algorithms incorporated in `Bagua` automatically benefit from our system optimizations, like memory management, execution management, communication/computation overlapping and so on, so that developers could take full advantage of the algorithm without a compromise caused by an inefficient implementation.

In this tutorial, we take Quantized Adam (Q-Adam) algorithm, inspired by this [paper](https://arxiv.org/pdf/2102.02888.pdf), as an example to describe how to create a new algorithm with Bagua. We also welcome contributions to add more builtin algorithms!

Let's first summarize the updating rule of Q-Adam algorithm as follows: ($w$ is the warm-up steps)

- Warm up stage: ($t < w$ )
  1. Calculating gradients $\textbf{g}_t^{(i)}$.
  2. Communicate $\textbf{g}_t^{(i)}$ from all workers with full precision to get $\textbf{g}_t$.
  3. Update $\textbf{m}_t$ and $\textbf{v}_t$: 
     - $\textbf{m}_t = \beta_1 \textbf{m}_{t-1} + (1-\beta_1)\textbf{g}_t$
     - $\textbf{v}_t = \beta_2 \textbf{v}_{t-1} + (1-\beta_2)\textbf{g}_t^2$

  4. Update model $\textbf{x}_t$:
     - $\textbf{x}_t = \textbf{x}_{t-1} - \gamma \frac{\textbf{m}_t}{\sqrt \textbf{v}_t+\epsilon}$


- Compression stage: ($t >= w$ )
  1. Calculating gradients $\textbf{g}_t^{(i)}$.
  2. Update $\textbf{m}_t$ with local gradients:
     - $\textbf{m}_t^{(i)} = \beta_1 \textbf{m}_{t-1}^{(i)} + (1-\beta_1)\textbf{g}_t^{(i)}$
  3. Compress $\textbf{m}_t^{(i)}$ into $\mathcal{C}(\textbf{m}_t^{(i)})$.
  4. Communicate $\mathcal{C}(\textbf{m}_t^{(i)})$ from all workers with low precision to get $\textbf{m}_t$
  5.  Update model $\textbf{x}_t$:
      - $\textbf{x}_t = \textbf{x}_{t-1} - \gamma \frac{\textbf{m}_t}{\sqrt \textbf{v}_w+\epsilon}$


To implement such an advanced distributed learning algorithm in any other popular ML system is far from trivial. Basically, the developer has to hack deeply into the source code and break their fine-grained communication optimizations. As the result, it is likely that one cannot observe any speedup compared with the basic Allreduce operation, actually in most cases it's even slower. 

Bagua provides a class called ``` Algorithm```. All a developer needs to do is to override pre-defined functions of this class as she wishes. (see [API document](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/algorithms/index.html#bagua.torch_api.algorithms.Algorithm) for more detailed information). In this example of Q-Adam, we need to override six functions as follows:

1. `__init__()`: Initializing the algorithm. Here Q-Adam algorithm requires an optimizer called `QAdamOptimizer`, which is a modified Adam optimizer available in Bagua. It also needs to know the warm-up steps defined by the user.

```python
def __init__(self, qadam_optimizer, warmup_step):
    self.optimizer = qadam_optimizer
    self.warmup_steps = warmup_steps
```

2. ```need_reset()```: As we can see, Q-Adam algorithm has two stages, which have very different logic regarding the communication contents and updating rules. ```need_reset()``` compares the current iteration with the warm-up steps, such that it can tell the `Bagua` backend to reset the algorithm. This function is checked by the Bagua engine for every iteration.

```python
def need_reset(self):
    return self.optimizer.step_id == self.warmup_steps
```

3. ```init_tensors()```: This function defines what needs to be communicated by registering intended tensors into the Bagua backend. Note that a developer can register any tensors as she wants. Q-Adam needs to communication gradients or momentums, therefore, we register them according to the current stage.

```python
tensors = []
for param, name in parameters:
    if self.optimizer.step_id < self.warmup_steps:
        registered_tensor = param.bagua_ensure_grad().to_bagua_tensor(name)
    else:
        registered_tensor = param.momentum.to_bagua_tensor(name)
    tensors.append(registered_tensor)
return tensors
```

4. ```tensors_to_buckets()```: This function is related to the tensor fusion optimization. Bagua would fuse small tensors into buckets to conduct the communication. In this function, one can customize which tensors should be fused together. By default, Bagua will fuse tensors based on the order of gradient computation during the backward.

5. ```init_operations()```: This function can define communication and computation operations of the algorithm. Let's first talk about the warm-up stage. Since we just need to average gradients, we adopt `append_centralized_synchronous_op` without compression, which is a centralized, full precision, synchronous communication operation. After the communication, updating $\textbf{m}$, $\textbf{v}$ and $\textbf{x}$ will take place locally in the `QAdamOptimizer.step()` after the backward process. In the compression stage, it becomes more complicated. As shown in the algorithm, we need to update $\textbf{m}$ before the communication. To support this process, we use `append_python_op` to add a python function `calculate_momentum` to momentum tensors. Then we can use `append_centralized_synchronous_op` with `MinMaxUInt8` compression to communicate momentums.


```python
if self.optimizer.step_id < self.warmup_steps:
    bucket.append_centralized_synchronous_op()
else:
    def calculate_momentum(*args):
        with torch.cuda.stream(_get_global_state().get_communication_stream()):
            beta1, beta2  = self.optimizer.param_groups[0]['betas']
            for tensor in bucket.tensors:
                tensor.mul_(beta1).add_(tensor._one_bit_grad, alpha=1 - beta1)

    bucket.append_python_op(calculate_momentum)
    bucket.append_centralized_synchronous_op(
        hierarchical=True,
        scattergather=True,
        compression="MinMaxUInt8",
    )

```

6. ```init_backward_hook()```: `Bagua` backend will triger this function for each tensor when its gradient calculation is finished. Then the algorithm is responsible to mark corresponding tensors as ready for executing the predefined operations in the previous step.

```python
def init_backward_hook(self, bagua_module: BaguaModule):
    def hook_momentum(parameter_name, parameter):
        parameter.momentum.bagua_mark_communication_ready()
    def hook_grad(parameter_name, parameter):
        parameter.grad.bagua_mark_communication_ready()
    return hook_grad if self.optimizer.step_id < self.warmup_steps else hook_momentum
```

Now we can use our newly defined algorithm in the training! To try out your algorithm, simply initialize our new algorithm in the training script and provide it to the `with_bagua` interface. Enjoy!

```python
optimizer = QAdamOptimizer(model.parameters())
algorithm = QAdamAlgorithm(optimizer, warmup_steps=100))
model.with_bagua([optimizer], algorithm=algorithm)
```














