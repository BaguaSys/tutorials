# Cached Dataset

When data loading is slow or data preprocessing is tedious, they could be the bottleneck of the whole training process. Bagua provides cached dataset to speedup this process by caching data samples in memory, so that reading these samples after the first time can be much faster.

## Usage

[`CachedDataset`](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/contrib/index.html#bagua.torch_api.contrib.CachedDataset) is
a Pytorch custom dataset (see [Creating a Custom Dataset for your files](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)).
It wraps a Pytorch dataset and caches its samples into a distributed key-value store. We can specify the backend to
use on the initialization of a cached dataset. Currently [Redis](https://redis.io/) is supported, which is an **in-memory** data store.

By default, cached dataset will spawn a new Redis instance on each worker node, and data is sharded across all
Redis instances on all nodes in the Bagua job. We can specify the maximum memory limit to use for each node, by passing
`capacity_per_node` to `CachedDataset`.

The following is an example to use a Redis-backend cached dataset, the maximum memory limit on each node is `400GB`. A
4-node Bagua job can have a maximum memory limit of `1.6TB`.

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

By setting `cluster_mode=False`, we can restrict each training node to use only its local Redis instance.

```python
cache_dataset = CachedDataset(
 dataset,
 backend="redis",
 dataset_name="ds",
 cluster_mode=False,
 capacity_per_node=400 * 1024 * 1024 * 1024,
)
```

We can also use existing Redis servers as the backend store by passing a list of host information of redis servers to `hosts`.

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

Multiple cached dataset share the same backend store, thus we need to specify a unique name for each dataset to avoid
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

cache_dataset2 = CachedDataset(
 dataset2,
 backend="redis",
 dataset_name="ds2",
 capacity_per_node=400 * 1024 * 1024 * 1024,
)
```

It should be noted that Redis instance will only be spawned once on each node, and the other cached dataset will reuse the existing Redis instance. Only parameters[^1] to spawn the first Redis instance will take effect. In the example above, the maximum memory for on each node will be `400GB` even if we set `capacity_per_node` to a different number when initializing `cache_dataset2`. 

[^1]: `cluster_mode` and `capacity_per_node` are used to spawn new Redis instances when `hosts=None`. See [RedisStore](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/contrib/utils/redis_store/index.html#bagua.torch_api.contrib.utils.redis_store.RedisStore)
for more information.

### Dataset with augmentation

For dataset with augmentation, we can not use cached dataset directly. Instead, we can define our own custom dataset
using [CachedLoader](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/contrib/index.html#bagua.torch_api.contrib.CacheLoader)[^2].
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

[^2]: `CachedDataset` is built upon `CacheLoader` as well.

## Benchmark result

On an important internal 3D mesh training task, where the data preprocessing becomes the major bottleneck, with one NVIDIA Tesla V100 GPU, **using cached loader can reduce the
end-to-end training time by more than 60%, only incurring a small overhead to write to the key-value store
in the first epoch**.

|                     | w/o Cached Loader   | w. Cached Loader    |
|---------------------|---------------------|---------------------|
| Epoch #1 Time (s)   |     6375            |     6473            |
| Epoch #2 Time (s)   |     6306            |     2264            |
| Epoch #3 Time (s)   |     6321            |     2240            |

