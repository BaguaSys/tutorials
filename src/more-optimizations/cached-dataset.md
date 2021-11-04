# Cached Dataset

When samples need tedious preprocessing to produce, or reading the dataset itself is slow, which could slow down the
whole training process. Bagua provides cached dataset to speedup this process by caching data samples in memory, so
that these samples after the first time can be much faster.

## Usage

[`CachedDataset`](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/contrib/index.html#bagua.torch_api.contrib.CachedDataset) is
a customized dataset, (see [Creating a Custom Dataset for your files](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)).
It wraps a Pytorch dataset and caches its samples into a distributed key-value store. We can specify the backend to
use on the initialization of a cached dataset. Currently [Redis](https://redis.io/) is supported, which is a **in-memory** data store.

By default, cached dataset will spawn a new Redis instance on each worker node, and data is sharded across all
Redis instances on all nodes in this Bagua job. We can specify the maximum memory limit to use for each node, by passing
`capacity_per_node` to `CachedDataset`.

The following is an example to use a Redis-backend cached dataset, the maximum memory limit on each node is `400GB`. A
4-node Bagua job will have a maximum memory limit of `1.6TB`.

```python
from bagua.torch_api.contrib import CachedDataset

cache_dataset = CachedDataset(
 dataset,
 backend="redis",
 dataset_name="ds",
 capacity_per_node=400 * 1024 * 1024 * 1024,
)
dataloader = torch.utils.data.DataLoader(cached_dataset)

for i, (input, target) in enumerate(dataloader):
    ...
```

By setting `cluster_mode=False`, we can restrict each training node using its local Redis instance.

```python
cache_dataset = CachedDataset(
 dataset,
 backend="redis",
 dataset_name="ds",
 cluster_mode=False,
 capacity_per_node=400 * 1024 * 1024 * 1024,
)
```

We can also use existing Redis servers as the backend store.

```python
hosts = [
    {"host": "192.168.1.0", "port": "7000"},
    {"host": "192.168.1.1", "port": "7000"},
]
cache_dataset = CachedDataset(
    dataset,
    backend="redis",
    dataset_name="ds",
    hosts=hosts,
)
```

### Multiple cached dataset

Multiple cached dataset shares the same backend store, thus we need to specify a unique name for each dataset to avoid
overwriting samples from each other.

```python
from bagua.torch_api.contrib import CachedDataset

dataset1 = ...
dataset2 = ...

cache_dataset1 = CachedDataset(
 dataset1,
 backend="redis",
 dataset_name="ds1",
 capacity_per_node=400 * 1024 * 1024 * 1024,
)

cache_dataset1 = CachedDataset(
 dataset2,
 backend="redis",
 dataset_name="ds2",
 capacity_per_node=400 * 1024 * 1024 * 1024,
)
```

It should be noted that the maximum memory for on each node is `400GB`, cached dataset will reuse the same Redis instance
on each node. Only parameters to spawn the first Redis instance will take effect.

### Dataset with augmentation

For dataset with augmentation, we can not use cached dataset directly. Instead, we can define our own custom dataset
using [CachedLoader](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/contrib/index.html#bagua.torch_api.contrib.CacheLoader).
Here is an example.

```python
import torch.utils.data as data
from bagua.torch_api.contrib import CacheLoader


class PanoHand(data.Dataset):
    def __init__(self):
        super(PanoHand, self).__init__()

        self.img_list = ...
        self.cache_loader = CacheLoader(
            backend="redis",
            capacity_per_node=400 * 1024 * 1024 * 1024,
            hosts=None,
            cluster_mode=True,
        )

    def __getitem__(self, idx):
        return self.get_training_sample(idx)

    def _process_fn(self, idx):
        # preprocessing to produce deterministic result
        ...

    def get_training_sample(self, idx):
        ret = self.cache_loader.get(idx, self._process_fn)

        # data augmentation
        ...


    def __len__(self):
        return len(self.img_list)

```

## Benchmark result


