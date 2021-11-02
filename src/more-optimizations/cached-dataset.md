# Cached Dataset

When samples need tedious preprocessing to produce, or reading the dataset itself is slow, which could slow down the
whole training process. Bagua provides cached dataset to speedup this process by caching data samples in memory, so
that these samples after the first time can be much faster. See
[API documentation](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/contrib/cached_dataset/index.html#bagua.torch_api.contrib.cached_dataset.CachedDataset) for details.


