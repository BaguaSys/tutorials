# More Optimizations

Besides communication algorithms, Bagua supports many convenient tools to
further accelerate your training workload. Currently we support:

1. [Generic fused optimizer](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/fuse_optimizer/index.html), which fuse optimizer step operations for multiple
layers, and it is generic because it can be applied to arbitrary PyTorch
optimizer, in contrast to
[apex](https://nvidia.github.io/apex/optimizers.html)'s approach, where only
some specific optimizers are implemented
2. [Load balanced data loader](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/contrib/data/load_balancing_data_loader/index.html), which accelerates workloads such as NLP and speech
   where training samples are of different length. This dataloader distribute
   training samples in a way that each worker receives samples of similar
   length, so that they finish a batch in similar time, mitigating the straggler
   problem in distributed setups.
   
We welcome more contributions!
