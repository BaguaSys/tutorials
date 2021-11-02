# Generic Fused Optimizer

Bagua provides generic fused optimizer which improve the performance of optimizers by fusing the optimizer `.step()` operation on multiple layers. It can be applied to arbitrary PyTorch optimizer. See [API documentation](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/contrib/index.html#bagua.torch_api.contrib.fuse_optimizer) for details.
