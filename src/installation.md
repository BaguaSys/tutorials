# Installation

To install Bagua, besides your deep learning framework (like [PyTorch](https://pytorch.org/get-started/locally/)
), you need the following libraries installed on your system:

* [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), with CUDA version >= 10.1
* [Rust Compiler](https://www.rust-lang.org/tools/install)

We provide an automatic installation script for Linux. Just run the following command to install all dependencies and Bagua:

```python
curl -Ls https://raw.githubusercontent.com/BaguaSys/bagua/master/install.sh | sudo bash
```

If you already installed the dependencies, you can install bagua python package with the following commands:

Install release version:

```shell
python3 -m pip install bagua
```

Install develop version:

```shell
python3 -m pip install git+https://github.com/BaguaSys/bagua.git
```
