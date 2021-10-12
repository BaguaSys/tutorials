# Bagua-Net

Bagua-Net is a low level communication acceleration feature provided by Bagua. It can greatly improve the throughput of AllReduce on TCP network.

Technically, Bagua-Net is a plugin for [NVIDIA NCCL communication library](https://developer.nvidia.com/nccl), the fastest generally avaiable GPU communication implementation now (2021). It replaces the TCP communication related logic in NCCL to futher improve the performance by a wide margin.

By enabling Bagua-Net, the communication efficiency can be increased by 83% ([code](https://github.com/BaguaSys/bagua/tree/master/examples/benchmark), and the end2end training throughput can be increased by 35%:

```
# VGG16 on 4x8xV100 NCCL default implementation
Running benchmark...
Iter #0: 2620.2 img/sec GPU
Iter #1: 2771.9 img/sec GPU
Iter #2: 2772.6 img/sec GPU
Iter #3: 2794.5 img/sec GPU
Iter #4: 2627.9 img/sec GPU
Iter #5: 2787.8 img/sec GPU
Iter #6: 2775.9 img/sec GPU
Iter #7: 2741.6 img/sec GPU
Iter #8: 2760.0 img/sec GPU
Iter #9: 2796.6 img/sec GPU
Img/sec per GPU: 85.8 +-3.8
Total img/sec on 32 GPU(s): 2744.9 +-122.3

# VGG16 on 4x8xV100 Bagua-Net enabled
Running benchmark...
Iter #0: 4081.0 img/sec GPU
Iter #1: 4072.0 img/sec GPU
Iter #2: 4106.4 img/sec GPU
Iter #3: 4081.7 img/sec GPU
Iter #4: 4064.8 img/sec GPU
Iter #5: 4122.1 img/sec GPU
Iter #6: 3857.7 img/sec GPU
Iter #7: 4128.3 img/sec GPU
Iter #8: 4125.5 img/sec GPU
Iter #9: 3826.6 img/sec GPU
Img/sec per GPU: 126.5 +-6.4
Total img/sec on 32 GPU(s): 4046.6 +-205.2
```

<!--
## Some test results

### 1. Performance comparison of Bagua-Net and NCCL-TCP under 100G TCP network

![](source/img/nccl-test_Bagua-Net_vs_NCCL-TCP.png)

> Thanks to the tensor fusion of the communication library. The actual communication packets will be larger than 10MB. In this range, Bagua-Net has better performance than NCCL-TCP. I have also done some experiments. When training a small network, Bagua-Net is no worse than NCCL-TCP.

### 2. Bagua-Net's acceleration effect on Bagua's different algorithms

![](source/img/bagua-net_accelerate_bagua_algorithms.png)

> The data comes from the real 128 V100 ImageNet training. The throughput increase brought by Bagua-Net is 11% to 68%.
-->

To enable Bagua-Net, you only need to pass the `--enable-bagua-net` argument in `bagua.distributed.launch` or `bagua.distributed.run`. No code change in your training script.

For example, with [this](https://github.com/BaguaSys/examples/blob/main/benchmark/synthetic_benchmark.py) distributed training example, you can launch the job with

```
python3 -m bagua.distributed.launch --enable-bagua-net \
    --nproc_per_node=8 synthetic_benchmark.py --algorithm gradient_allreduce
```

> It worth noting that you can even use `bagua.distributed.launch` or `bagua.distributed.run` with `--enable-bagua-net` argument to launch PyTorch-DDP jobs to improve the training throughput without migrating your code to Bagua.

<!-- 
## Enable Bagua-Net


```bash
# Install Bagua-Net
git clone https://github.com/BaguaSys/bagua.git
cd bagua/rust/bagua-net/cc && make
export BAGUA_NET_LIBRARY_PATH=$(readlink -f .)

# Install nccl and nccl-test
git clone https://github.com/NVIDIA/nccl.git && cd nccl && git checkout v2.10.3-1
make -j src.build && make install
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1

# Run nccl-test NCCL-TCP
mpirun \
  --allow-run-as-root \
  -H ${HOST1}:1,${HOST2}:1 --np 2 \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include eth01 \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1

# Run nccl-test with bagua-net
mpirun \
  --allow-run-as-root \
  -H ${HOST1}:1,${HOST2}:1 --np 2 \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include eth01 \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$BAGUA_NET_LIBRARY_PATH \
    ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
# If the installation is successful, there will be a log like this `NCCL INFO Using network BaguaNet`.
```
-->
