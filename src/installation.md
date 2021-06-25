# Installation

## Install locally

To install Bagua, besides your deep learning framework (like [PyTorch](https://pytorch.org/get-started/locally/)), you need the following dependencies installed on your system:

* [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), with CUDA version >= 10.1
* [Rust Compiler](https://www.rust-lang.org/tools/install)
* MPI >= 3.0, for example [Open MPI](https://www.open-mpi.org/)
* [hwloc](https://www.open-mpi.org/projects/hwloc/) >= 2.0
* [CMake](https://cmake.org/) >= 3.19

We provide an automatic installation script for Ubuntu. Just run the following command to install Bagua and above libraries (except for CUDA, you should always install CUDA by yourself):

```python
curl -Ls https://raw.githubusercontent.com/BaguaSys/bagua/master/install.sh | sudo bash
```

If you already have dependencies installed on your system, you can install the bagua python package only:

```shell
# release version
python3 -m pip install bagua -f https://repo.arrayfire.com/python/wheels/3.8.0/

# develop version (git master)
python3 -m pip install git+https://github.com/BaguaSys/bagua.git -f https://repo.arrayfire.com/python/wheels/3.8.0/
```

## Use Docker image

We provide Docker image with Bagua installed based on official PyTorch images. You can find them on [DockerHub](https://hub.docker.com/r/baguasys/bagua).
