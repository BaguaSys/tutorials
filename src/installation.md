# Installation

To install Bagua, besides your deep learning framework (like [PyTorch](https://pytorch.org/get-started/locally/)), you need the following libraries installed on your system:

* [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), with CUDA version >= 10.1
* [Rust Compiler](https://www.rust-lang.org/tools/install)

We provide an automatic installation script for Linux. Just run the following command to install above libraries and Bagua:

```python
curl -Ls https://raw.githubusercontent.com/BaguaSys/bagua/master/install.sh | sudo bash
```

If you already have required libraries installed on your system, you can install bagua python package with the following commands:

Install release version:

```shell
python3 -m pip install bagua -f https://repo.arrayfire.com/python/wheels/3.8.0/
```

Install develop version:

```shell
python3 -m pip install git+https://github.com/BaguaSys/bagua.git -f https://repo.arrayfire.com/python/wheels/3.8.0/
```
