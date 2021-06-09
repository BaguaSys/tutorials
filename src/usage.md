# Prepare the Code

To use `Bagua`, you need make the following change on your training code:

First, import bagua and make your script distributed aware:

```python
import bagua.torch_api as bagua
torch.cuda.set_device(bagua.get_local_rank())
bagua.init_process_group()
```

Then, use torch's distributed sampler for your data loader:

```python
train_dataset = ...
test_dataset = ...

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
    num_replicas=bagua.get_world_size(), rank=bagua.get_rank())

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
)

test_loader = torch.utils.data.DataLoader(test_dataset, ...)
```

Finally, wrap you model and optimizer with bagua by adding one line of code to your original script:

```python
# define your model and optimizer
model = ...
model = model.cuda()
optimizer = ...

# wrap your model and optimizer with bagua
model, optimizer = bagua.bagua_init(
    model, optimizer, distributed_algorithm="allreduce"
)
```

More examples can be found in [here](https://github.com/BaguaSys/examples).