# PREREQUISITES

**Linux Distributions**

[Bagua](https://github.com/BaguaSys/Bagua) is tested on the following:
* [CentOS](https://www.centos.org/)
* [Ubuntu](https://www.ubuntu.com/)

**CUDA**

To install Bagua, make sure you have a [CUDA-capable](https://developer.nvidia.com/zh-cn/cuda-zone) system, and install the CUDA version suited to your machine. Often, the latest CUDA version is better.

**Python**

For many popular Linux distributions, Python 3.6 or greater is installed by default, which meets our recommendation. Otherwise, use APT to install Python:

```shell
apt install python
```

or install it from [Python website](https://www.python.org/downloads/).

**Pytorch**

There are multiple ways to install Pytorch. To install it from pip:

```shell
pip install pytorch
```

Other choices can be found in [Pytorch documentations](https://pytorch.org/get-started/locally/).

**Rust**

To install Bagua, you will need install [Rust](https://www.rust-lang.org/). Run the following command to download and install Rust.

```shell
curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y
```


# INSTALLATION

# BUILDING FROM SOURCE