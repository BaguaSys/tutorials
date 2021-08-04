# Quick-start on AWS Cloud

We provide Bagua image with [Amazon Machine Image (AMI)](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AMIs.html) on AWS cloud so that one can easily try Bagua system on EC2 machines.

## How to Use Bagua AMI


| Bagua version  | AMI ID | Published Time |
|---|---|---|
| 0.6.3 | ami-07d61b187eb536338 | July 30, 2021 |

If you want to use one instance with multiple GPUs, you can simply search and choose our AMI as your image in the first step of "launch instances". The Bagua AMI can be found with a unique AMI ID provided in this page.

If you want to use multiple instances with multiple GPUs, you need to set up a EC2 cluster first. There are many tools helping to create and maintain the EC2 cluster, such as [StarCluster](http://star.mit.edu/cluster/docs/latest/index.html). After configuring of StarCluster, you can easily create a cluster with one command in your terminal. Besides, the created cluster is automatically configured with NFS shared home folder and passwordless ssh connection with each other, which would make it much more efficient to use and manage the cluster. For more details about StarCluster, please follow their [tutorial](http://star.mit.edu/cluster/docs/latest/quickstart.html). To use Bagua AMI within StarCluster, you just need to fill in our AMI ID in the config file item **NODE_IMAGE_ID**.


## Results on AWS