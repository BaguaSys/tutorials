# Generic Fused Optimizer

Bagua provides a generic fused optimizer which improves the performance of optimizers by fusing the optimizer `.step()`
operation on multiple layers. It can be applied to any PyTorch optimizer.

## Usage

To use the fused optimizer given an existing PyTorch optimizer, do:

* convert the optimizer to a fused optimizer using `bagua.contrib.fuse_optimizer()`,
* perform optimizer step operation with the `.fuse_step()` method.

Here is an example of how to use fused optimizer with a popular custom `BertAdam` optimizer (any other PyTorch optimizers including PyTorch built-in optimizers will also work):

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

### Integration with Bagua distributed communication algorithms

Fused optimizer and Bagua module both reset the storage pointer of parameters to flatten tensors by default. In order to perform a
more effective fused update, it is recommended to disable tensor flatten in
[`with_bagua`](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/distributed/index.html#bagua.torch_api.distributed.BaguaModule.with_bagua)
by setting its `do_flatten` parameter to `False`:

```python
model = model.with_bagua([optimizer], do_flatten=False)
```
### Saving and loading optimizer `state_dict`

A fused optimizer behaves just like an ordinary optimizer except that it is enabled to perform fused parameter updates with `.fuse_step()` method.
We can store and load its `state_dict` in the original ways (see [Saving & Loading a General Checkpoint for Inference and/or Resuming Training](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training)).

#### Save
```python
optimizer = bagua.contrib.fuse_optimizer(optimizer, do_flatten=True)

torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    },
    ...
    PATH,
)
```

#### Load
```python
optimizer = TheOptimizerClass(*args, **kwargs)

# load optimizer state first
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# convert to a fused optimizer before training
optimizer = bagua.contrib.fuse_optimizer(optimizer, do_flatten=True)
```


## Benchmark

On BERT base model, with 8 NVIDIA Tesla
V100 GPUs. The benchmark result shows that **by enabling generic fused optimizer, the end-to-end training time can be reduced by 17%**.

|                     | w/o Fused Optimizer | w. Fused Optimizer  |
|---------------------|---------------------|---------------------|
| Epoch Time (s)      |   3252              |     2767            |
| Accuracy            |   0.9254            |     0.9260          |
