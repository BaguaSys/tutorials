# Kubernetes operator for Bagua jobs

Bagua supports kubernetes with a dedicated [Bagua operator](https://github.com/BaguaSys/operator). This greatly simplifies deployments in modern computing cluster.

## Prerequisites

- [Kubernetes](https://kubernetes.io/) >= 1.16

## Installation

### Run the operator locally

```shell
git clone https://github.com/BaguaSys/operator.git
cd operator

# install crd
kubectl apply -f config/crd/bases/bagua.kuaishou.com_baguas.yaml

go run ./main.go
```

### Deploy the operator

Install Bagua on an existing Kubernetes cluster.

```shell
kubectl apply -f https://raw.githubusercontent.com/BaguaSys/operator/master/deploy/deployment.yaml
```

Enjoy! Bagua will create resources in namespace `bagua`.

## Examples

You can run demos in `config/samples`:

### Static mode

"Static mode" means running the Bagua distributed training job with fixed number of nodes, and no fault tolerance.

```shell
kubectl apply -f config/samples/bagua_v1alpha1_bagua_static.yaml
```

Verify pods are running:

```yaml
kubectl get pods

NAME                           READY   STATUS    RESTARTS   AGE
bagua-sample-static-master-0   1/1     Running   0          45s
bagua-sample-static-worker-0   1/1     Running   0          45s
bagua-sample-static-worker-1   1/1     Running   0          45s
```

### Elastic mode

"Elastic mode" means running the Bagua distributed training job in [elastic mode](https://baguasys.github.io/tutorials/elastic-training/index.html), which means the number of nodes can be dynamically adjusted, and the job is fault tolerant.

```shell
kubectl apply -f config/samples/bagua_v1alpha1_bagua_elastic.yaml
```

Verify pods are running

```yaml
kubectl get pods

NAME                            READY   STATUS    RESTARTS   AGE
bagua-sample-elastic-etcd-0     1/1     Running   0          63s
bagua-sample-elastic-worker-0   1/1     Running   0          63s
bagua-sample-elastic-worker-1   1/1     Running   0          63s
bagua-sample-elastic-worker-2   1/1     Running   0          63s
```
