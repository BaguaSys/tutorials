# How to Create a New Algorithm

Thanks to the innovative design of Bagua, algorithm developers now can easily create, test and benchmark their distributed learning algorithms in a realistic system. Within Bagua, developers have the freedom to manipulate almost all the details regarding the data-parallel distributed training, including What to communicate, When to communicate, How to update the model and so on. Besides, algorithms incorporated in Bagua automatically benefit from our system optimizations, like memory management, execution management, communication/computation overlapping and so on, so that developers could take full advantage of the algorithm without a compromise caused by an inefficient implementation.

In this tutorial, we take Quantized Adam (Q-Adam) algorithm, inspired by this [paper](https://arxiv.org/pdf/2102.02888.pdf), as an example to describe how to create a new algorithm with Bagua. The complete code can be found [here](https://github.com/BaguaSys/bagua/blob/master/bagua/torch_api/algorithms/q_adam.py). We also welcome contributions to add more built-in algorithms!

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

Bagua provides two base classes: `Algorithm` and `AlgorithmImpl`. The former is used to declare an algorithm, including all parameters an algorithm needs. The latter is used to actually implement the algorithm. When an end user uses an algorithm, she provides the algorithm declaration to Bagua, and Bagua will reify the algorithm implementation instance to actually run the algorithm[^1]. In this example of Q-Adam, we implement the algorithm as follows:

[^1]: In this way, an end user can pass a single algorithm declaration to multiple models, in order to run the same algorithm on multiple models.

**Create a `QAdamAlgorithm` class that inherits `Algorithm`:**

1. `__init__()`: Define the parameters that  `QAdamAlgorithm` needs.

```python
class QAdamAlgorithm(Algorithm):
    def __init__(self, q_adam_optimizer: QAdamOptimizer, hierarchical: bool = True):
        self.hierarchical = hierarchical
        self.optimizer = q_adam_optimizer
```

2. `reify()`:  Create and return a  ``` QAdamAlgorithmImpl``` instance.

```python
 def reify(self, process_group: BaguaProcessGroup) -> QAdamAlgorithmImpl:
        return QAdamAlgorithmImpl(
            process_group,
            q_adam_optimizer=self.optimizer,
            hierarchical=self.hierarchical,
        )
```

**Create a ``` QAdamAlgorithmImpl``` class that inherits ``` AlgorithmImpl```  [^2]:**

1. `__init__()`: Initializing the algorithm. Here Q-Adam algorithm requires an optimizer called `QAdamOptimizer`, which is a specifically customized optimizer based on the Adam optimizer in order to meet the special updating rule of Q-Adam algorithm. Compared with the original Adam optimizer, the main difference of `QAdamOptimizer` is that, in the compression stage, communicating and updating $\textbf{m}$ are conducted by the Bagua backend, instead of the optimizer. Like all other optimizers in PyTorch, `QAdamOptimizer` needs to be initialized with model parameters. Besides, an extra argument `warmup_steps` decides how many steps of the warm-up stage. `QAdamAlgorithm` can be initialized simply by the `QAdamOptimizer`. 

```python
from bagua.torch_api.algorithms import q_adam 
optimizer = q_adam.QAdamOptimizer(
    model.parameters(),
    lr=1e-3,
    warmup_steps=100
)
```


```python
def __init__(
        self,
        process_group: BaguaProcessGroup,
        q_adam_optimizer: QAdamOptimizer,
        hierarchical: bool = True,
    ):
        super(QAdamAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.optimizer = q_adam_optimizer
        self.warmup_steps = self.optimizer.warmup_steps
```


[^2]: All ```AlgorithmImpl``` instances must initialize the ```process_group``` to work on on initialization. See [Bagua API Documentation](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/algorithms/base/index.html) for details.

2. ```need_reset()```: As we can see, Q-Adam algorithm has two stages, which have very different logic regarding the communication contents and updating rules. ```need_reset()``` compares the current iteration with the warm-up steps, such that it can tell the `Bagua` backend to reset the algorithm. This function is checked by the Bagua engine for every iteration.

```python
def need_reset(self):
    return self.optimizer.step_id == self.warmup_steps
```

3. ```init_tensors()```: This function defines what needs to be communicated by registering intended tensors into the Bagua backend. Note that a developer can register any tensors as she wants. Q-Adam needs to communicate gradients or momentums, therefore, we register them according to the current stage.

```python
tensors = []
for param, name in parameters:
    if self.optimizer.step_id < self.warmup_steps:
        registered_tensor = param.bagua_ensure_grad().to_bagua_tensor(name, bagua_module.bagua_module_name)
    else:
        registered_tensor = param.momentum.to_bagua_tensor(name, bagua_module.bagua_module_name)
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

6. ```init_backward_hook()```: `Bagua` backend will trigger this function for each tensor when its gradient calculation is finished. Then the algorithm is responsible to mark corresponding tensors as ready for executing the predefined operations in the previous step.

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
optimizer = QAdamOptimizer(
    model.parameters(),
    lr=1e-3,
    warmup_steps=100
)
algorithm = QAdamAlgorithm(optimizer))
model.with_bagua([optimizer], algorithm=algorithm)
```