# Installation

## Installing Bagua

Wheels (precompiled binary packages) are available for Linux (x86_64). Package names are different depending on your CUDA Toolkit version (CUDA Toolkit version is shown in `nvcc --version`).

| CUDA Toolkit version | Installation command        |
|----------------------|-----------------------------|
| >= v10.2             | `pip install bagua-cuda102` |
| >= v11.1             | `pip install bagua-cuda111` |
| >= v11.3             | `pip install bagua-cuda113` |

Add `--pre` to `pip install` commands to install pre-release (development) versions.

## Install from source

To install Bagua by compiling from source code on your machine, you need the following dependencies installed on your system:

* [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), with CUDA version >= 10.1
* [Rust Compiler](https://www.rust-lang.org/tools/install)
* MPI >= 3.0, for example [Open MPI](https://www.open-mpi.org/)
* [hwloc](https://www.open-mpi.org/projects/hwloc/) >= 2.0
* [CMake](https://cmake.org/) >= 3.17

We provide an automatic installation script for Ubuntu. Just run the following command to install Bagua and above libraries (except for CUDA, you should always install CUDA by yourself):

```python
curl -Ls https://raw.githubusercontent.com/BaguaSys/bagua/master/install.sh | sudo bash
```

Run the following commands to install Bagua (source code packages, which will be compiled on your machine).

```bash
# release version
python3 -m pip install bagua --upgrade

# develop version (git master)
python3 -m pip install --pre bagua --upgrade
```

## Use Docker image

We provide Docker image with Bagua installed based on official PyTorch images. You can find them on [DockerHub](https://hub.docker.com/r/baguasys/bagua).
