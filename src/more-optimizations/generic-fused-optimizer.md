# Generic Fused Optimizer

Bagua provides generic fused optimizer which improve the performance of optimizers by fusing the optimizer `.step()`
operation on multiple layers. It can be applied to arbitrary PyTorch optimizer.

## Usage

To use fused optimizer, we need:
* convert the optimizer to a fused optimizer
* perform fused parameter update

Here is an example of how to use fused optimizer with [`BertAdam`](https://huggingface.co/transformers/main_classes/optimizer_schedules.html?highlight=adamw#transformers.AdamW).

```python
from pytorch_pretrained_bert.optimization import BertAdam
import bagua.torch_api as bagua

optimizer = BertAdam(parameters, ...)
optimizer = bagua.contrib.fuse_optimizer(optimizer, do_flatten=True)

for batch, input in enumerate(trainloader):
    model.zero_grad()
    ...
    optimizer.fuse_step()
```

See [API documentation](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/contrib/index.html#bagua.torch_api.contrib.fuse_optimizer) for more details.

### Work with Bagua Module

Fused optimizer and Bagua module both will reset data pointers of module parameters by default. In order to perform a
more effective fused parameter update, we need to disable bucket flattening in
[`with_bagua`](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/distributed/index.html#bagua.torch_api.distributed.BaguaModule.with_bagua)
by setting its `do_flatten` to False.

```python
model = model.with_bagua([optimizer], do_flatten=False)
```

## Benchmark Result

We tested on a Kwai proprietary natural language processing dataset and BERT-BASE finetune model, with 8 NVIDIA Tesla
V100 GPUs. The result shows that **by enabling generic fused optimizer, the end-to-end training performance can be increased by 10%**.