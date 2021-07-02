# Quantized Adam (QAdam)

## Overview

QAdam is a communication compression algorithm that is specifically intended for Adam optimizer. Although there are lots of SGD-based gradients compression algorithms, e.g., QSGD, 1-bit SGD and so on, none of them can be directly applied to Adam optimizer because Adam is non-linearly dependent on the gradient. Empirical study also shows that Adam with gradient compression could suffer an obvious drop in the training accuracy and cannot converge to the same level as its non-compression counterpart. Motivated by this observation, we proposed QAdam based on this [paper](https://arxiv.org/pdf/2102.02888.pdf) to make it possible for Adam to benefit from communication compression.

## QAdam algorithm

Let's first have a look of the updating strategy of the original Adam, which can be summaried as:

$\textbf{m}_t = \beta_1 \textbf{m}_{t-1} + (1-\beta_1)\textbf{g}_t$

$\textbf{v}_t = \beta_2 \textbf{v}_{t-1} + (1-\beta_2)\textbf{g}_t^2$

$\textbf{x}_t = \textbf{x}_{t-1} - \gamma \frac{\textbf{m}_t}{\sqrt \textbf{v}_t+\epsilon}$

where $t$ is the index of iteration, $\textbf{x}$ represents model parameters, $\gamma$ is the learning rate, $\textbf{g}_t$ is gradient at step $t$.

As we discussed above, direct compression ${g}_t$ will lead to the diverge of training because of the non-linear component $\textbf{v}_t$. The intuition of QAdam is that $\textbf{v}$ tends to be very stable after a few epochs in the beginning, so we can set $\textbf{v}$ as constant afterward and only update $\textbf{m}$. Without the effect of $\textbf{v}$, we can compress $\textbf{m}$ without worrying about the drop of training accuracy.

Therefore, QAdam algorithm consists of two stages: warmup stage and compression stage. 

- **In the warmup stage** (usually takas 20% of the total iterations in the beginning), all workers communicate to average local gradients before updating $\textbf{m}$ and $\textbf{v}$ without compression. 
- **In the compression stage**, $\textbf{v}$ is frozen and not updated anymore. All workers update $\textbf{m}$ with its local gradients and compress it into $\mathcal{C}(\textbf{m})$. Then $\mathcal{C}(\textbf{m})$ will be communicated among workers.

A detailed description and analysis of the algorithm can be found in the [paper](https://arxiv.org/pdf/2102.02888.pdf).

## Benchmark

We provide some benchmark results [here](../benchmark/index.md) to compare the performance of QAdam of Bagua with other SOTA systems on BERT-Large finetune task.


## Limitation
As we discussed above, the QAdam is based on an assumption that the value of $\textbf{v}$ in Adam will quickly get stable after a few epochs of training. However, it may not work if this assumption breaks. Although we have tested the correctness of QAdam on BERT-Large, BERT-Base, ResNet50 and Deep Convolutional Generative Adversarial Networks, it is still possible that QAdam may fail on some other tasks. The condition of QAdam is still an interesting open problem.


## Example

To use QAdam algorithm, you first need to initialize a QAdam optimizer, which is similar as any other optimizers in PyTorch. After the initialization of ```QAdamOptimizer``` and ```QAdamAlgorithm```, simply putting them into the ```with_bagua()``` function of model.

```python
from bagua.torch_api.algorithms.q_adam import QAdamAlgorithm, QAdamOptimizer

optimizer = QAdamOptimizer(model.parameters(), warmup_steps = 100)
algorithm = QAdamAlgorithm(optimizer, hierarchical_reduce=True)
```

Then decorate your model with:

```python
model = model.with_bagua([optimizer], algorithm)
```