# DGX Spark Networking

The following guide starts with a two-node cluster, but it is also applicable to larger clusters.

See [this post](https://forums.developer.nvidia.com/t/6x-spark-setup/354399/56) for an example of 6-8 node Spark cluster.
Please keep in mind that tensor-parallel vLLM deployments usually work best with a number of nodes that corresponds to a power of 2, such as 2, 4, or 8 nodes. A 3-node mesh is mainly useful for pipeline parallelism or data parallelism.

The guide assumes that the nodes are named `spark` and `spark2`, but you can use any names.
Same with IP addresses: we use `192.168.177.0/24` subnet with `.11` and `.12` assigned to both nodes, but you can use any IP addresses, as long as they are in the same subnet.

For four Sparks connected without a QSFP switch, see the
[four-node ring example](#example-four-node-ring) below.

## DGX Spark ConnectX quirks

DGX Spark has a pretty unique ConnectX setup.

To achieve 200G transfer speed, ConnectX NIC needs ~x8 PCIe 5.0 lanes.

However, DGX Spark SOC can't provide more than x4 PCIe lanes per device due to hardware limitations.
So to achieve 200G on a single cable connection, each physical port shares the same pair of PCIe5 x4 connections.
Each PCIe 5 x4 link is represented by two Ethernet and two RoCE interfaces:

```bash
eugr@spark:~$ ibdev2netdev
rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
```

In this case, the single cable is plugged in the outermost QSFP port (the right one if looking from the back).
This port has two pairs of "twins" associated with it:

- Ethernet: `enp1s0f1np1` and `enP2p1s0f1np1`
- RoCE/IB: `rocep1s0f1` and `roceP2p1s0f1`

Each of the twins represents one PCIe x4 link and can provide up to 100G link speed.

For vLLM, we need RDMA over RoCE, so Ethernet speed is not that important, that's why we can assign IP only to one of the ports - in this case `enp1s0f1np1`.
However, in order to get full bandwidth in NCCL RDMA mode, we need to utilize **both** RoCE twins. It is achieved by setting `NCCL_IB_HCA` to both RoCE interfaces: `export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`

`./launch-cluster.sh` does this automatically, along with autodiscovery of interfaces, so as long as you set up your Ethernet interface properly, vLLM will utilize both RoCE twins.

Also, note that connecting two Sparks using **both** ports won't give you any noticeable advantage in bandwidth, so single connection is sufficient.
If you connect 3 Sparks by daisy-chaining them, you will only be able to sustain 100G between each pair of Sparks.

## Connecting 3 Sparks in a mesh cluster without a switch

Three Sparks can be connected together in a cluster without using a separate RoCE switch.
However, all three Sparks need to be on the same wired network using their 10G Ethernet ports (RJ-45, not QSFP). Being on the same wireless network should work too, but it's not recommended and was not tested.

You need to make sure they are connected the following way: port 0 on one Spark should connect to port 1 on another Spark (unlike non-mesh configuration).
Example diagram:

```mermaid
block-beta
    columns 1
    
    block:Spark3
        columns 2
        Title3["Spark 3"]:2
        s3p0["Port 0<br>192.168.187.13<br>192.168.188.13"] s3p1["Port 1<br>192.168.197.13<br>192.168.198.13"]
    end
    
    space
    
    block:Spark2
        columns 2
        Title2["Spark 2"]:2
        s2p0["Port 0<br>192.168.197.12<br>192.168.198.12"] s2p1["Port 1<br>192.168.177.12<br>192.168.178.13"]
    end
    
    space
    
    block:Spark1
        columns 2
        Title1["Spark 1"]:2
        s1p0["Port 0<br>192.168.177.11<br>192.168.178.11"] s1p1["Port 1<br>192.168.187.11<br>192.168.188.11"]
    end

    s1p0 <--> s2p1
    s2p0 <--> s3p1
    s3p0 <--> s1p1
```

## Connecting more than 2 Sparks in the cluster using a switch

For a switch-connected cluster, use a suitable QSFP/RoCE switch, for example [Microtik CRS812-DDQ](https://mikrotik.com/product/crs812_ddq) or [Mikrotik CRS804-DDQ](https://mikrotik.com/product/crs804_ddq).
Please refer to [this post](https://forums.developer.nvidia.com/t/6x-spark-setup/354399/56) for an example of setting up a 6-8 node Spark cluster.

## Network setup

### For dual Sparks or multiple Sparks using a QSFP switch

Assuming both are connected using rightmost QSFP port (when looking from the back).

Create `/etc/netplan/40-cx7.yaml` on `spark`:
```yaml
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      dhcp4: no
      dhcp6: no        # Explicitly disable DHCPv6
      link-local: []   # Restrict link-local addresses to static IPv4 only
      mtu: 9000
      addresses: [192.168.177.11/24]
    enP2p1s0f1np1:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [192.168.178.11/24]
```

Create `/etc/netplan/40-cx7.yaml` on `spark2`:
```yaml
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      dhcp4: no
      dhcp6: no        # Explicitly disable DHCPv6
      link-local: []   # Restrict link-local addresses to static IPv4 only
      mtu: 9000
      addresses: [192.168.177.12/24]
    enP2p1s0f1np1:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [192.168.178.12/24]
```

**DO NOT use the same subnet on both "twins"** - it will confuse autodiscovery and mess up routing.

Then run on each node:

```bash
sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```

Set up passwordless ssh. On spark:

```bash
wget https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/connect-two-sparks/assets/discover-sparks
chmod +x discover-sparks
./discover-sparks
```

MTU setting (testing):

```bash
sudo ip link set dev enp1s0f1np1 mtu 9000
```

### For 3-node mesh

3-node mesh is configured differently than dual clusters or clusters using a QSFP switch.

Assuming, your Sparks are connected according to the diagram above:

Create `/etc/netplan/40-cx7.yaml` on `spark1`:
```yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      dhcp4: no
      dhcp6: no        # Explicitly disable DHCPv6
      link-local: []   # Restrict link-local addresses to static IPv4 only
      mtu: 9000
      addresses: [192.168.177.11/24]
    enP2p1s0f0np0:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [192.168.178.11/24]
    enp1s0f1np1:
      dhcp4: no
      dhcp6: no        # Explicitly disable DHCPv6
      link-local: []   # Restrict link-local addresses to static IPv4 only
      mtu: 9000
      addresses: [192.168.187.11/24]
    enP2p1s0f1np1:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [192.168.188.11/24]
```

Create `/etc/netplan/40-cx7.yaml` on `spark2`:
```yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      dhcp4: no
      dhcp6: no        # Explicitly disable DHCPv6
      link-local: []   # Restrict link-local addresses to static IPv4 only
      mtu: 9000
      addresses: [192.168.197.12/24]
    enP2p1s0f0np0:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [192.168.198.12/24]
    enp1s0f1np1:
      dhcp4: no
      dhcp6: no        # Explicitly disable DHCPv6
      link-local: []   # Restrict link-local addresses to static IPv4 only
      mtu: 9000
      addresses: [192.168.177.12/24]
    enP2p1s0f1np1:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [192.168.178.12/24]
```

Create `/etc/netplan/40-cx7.yaml` on `spark3`:
```yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      dhcp4: no
      dhcp6: no        # Explicitly disable DHCPv6
      link-local: []   # Restrict link-local addresses to static IPv4 only
      mtu: 9000
      addresses: [192.168.187.13/24]
    enP2p1s0f0np0:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [192.168.188.13/24]
    enp1s0f1np1:
      dhcp4: no
      dhcp6: no        # Explicitly disable DHCPv6
      link-local: []   # Restrict link-local addresses to static IPv4 only
      mtu: 9000
      addresses: [192.168.197.13/24]
    enP2p1s0f1np1:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [192.168.198.13/24]
```

Then run (on each Spark):

```bash
sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```

### Passwordless SSH and benchmarks

Set up passwordless ssh. On the first spark:

```bash
wget https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/connect-two-sparks/assets/discover-sparks
chmod +x discover-sparks
./discover-sparks
```

**Benchmark connection (use perftest package):**

Run the receiver on `spark2` node:

```bash
ib_write_bw -d rocep1s0f1 --report_gbits -q 4 -R --force-link IB
```

Then run on `spark`:

```bash
ib_write_bw 192.168.177.12 -d rocep1s0f1 --report_gbits -q 4 -R --force-link IB
```

```
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF          Device         : rocep1s0f1
 Number of qps   : 4            Transport type : IB
 Connection type : RC           Using SRQ      : OFF
 PCIe relax order: ON
 ibv_wr* API     : ON
 TX depth        : 128
 CQ Moderation   : 1
 Mtu             : 1024[B]
 Link type       : IB
 Max inline data : 0[B]
 rdma_cm QPs     : ON
 Data ex. method : rdma_cm
---------------------------------------------------------------------------------------
 local address: LID 0000 QPN 0x03ec PSN 0xb680ae
 local address: LID 0000 QPN 0x03ed PSN 0x808800
 local address: LID 0000 QPN 0x03ee PSN 0x5b694a
 local address: LID 0000 QPN 0x03ef PSN 0xe2efd1
 remote address: LID 0000 QPN 0x03eb PSN 0x75f6ee
 remote address: LID 0000 QPN 0x03ec PSN 0x436140
 remote address: LID 0000 QPN 0x03ed PSN 0x81698a
 remote address: LID 0000 QPN 0x03ee PSN 0x4a8b11
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
 65536      20000            111.72             111.71             0.213070
---------------------------------------------------------------------------------------
```

**Latency test:**

Run the receiver on `spark2` node:

```bash
ib_write_lat -d rocep1s0f1 --report_gbits -R --force-link IB
```

Then run on `spark`:

```bash
ib_write_lat 192.168.177.12 -d rocep1s0f1 --report_gbits -R --force-link IB
```

```
---------------------------------------------------------------------------------------
                    RDMA_Write Latency Test
 Dual-port       : OFF          Device         : rocep1s0f1
 Number of qps   : 1            Transport type : IB
 Connection type : RC           Using SRQ      : OFF
 PCIe relax order: OFF
 ibv_wr* API     : ON
 TX depth        : 1
 Mtu             : 1024[B]
 Link type       : IB
 Max inline data : 220[B]
 rdma_cm QPs     : ON
 Data ex. method : rdma_cm
---------------------------------------------------------------------------------------
 local address: LID 0000 QPN 0x02ee PSN 0xb0c21c
 remote address: LID 0000 QPN 0x02ee PSN 0x14568b
---------------------------------------------------------------------------------------
 #bytes #iterations    t_min[usec]    t_max[usec]  t_typical[usec]    t_avg[usec]    t_stdev[usec]   99% percentile[usec]   99.9% percentile[usec]
 2       1000          1.42           1.93         1.47                1.47             0.00            1.57                    1.93
---------------------------------------------------------------------------------------
```

## NCCL Tests

### Dual Sparks or Sparks via QSFP switch

From https://build.nvidia.com/spark/nccl/stacked-sparks

```bash
# Install dependencies and build NCCL
sudo apt-get update && sudo apt-get install -y libopenmpi-dev
git clone -b v2.30u1 https://github.com/NVIDIA/nccl.git ~/nccl/
cd ~/nccl/
make -j src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"

# Set environment variables
export CUDA_HOME="/usr/local/cuda"
export MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
export NCCL_HOME="$HOME/nccl/build/"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH"
```

Build NCCL Test Suite:

```bash
# Clone and build NCCL tests
git clone https://github.com/NVIDIA/nccl-tests.git ~/nccl-tests/
cd ~/nccl-tests/
make MPI=1
```

Test on both nodes:

```bash
# Set network interface environment variables (use your active interface)
export UCX_NET_DEVICES=enp1s0f1np1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export OMPI_MCA_btl_tcp_if_include=enp1s0f1np1
export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
export NCCL_IB_DISABLE=0

# Run the all_gather performance test across both nodes
mpirun -np 2 -H 192.168.177.11:1,192.168.177.12:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  $HOME/nccl-tests/build/all_gather_perf -b 16G -e 16G -f 2

```

### 3-node mesh

```bash
# Install dependencies and build NCCL
sudo apt-get update && sudo apt-get install -y libopenmpi-dev
git clone -b v2.30u1 https://github.com/NVIDIA/nccl.git ~/nccl/
cd ~/nccl/
make -j src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"

# Set environment variables
export CUDA_HOME="/usr/local/cuda"
export MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
export NCCL_HOME="$HOME/nccl/build/"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH"
```

Build NCCL Test Suite:

```bash
# Clone and build NCCL tests
git clone https://github.com/NVIDIA/nccl-tests.git ~/nccl-tests/
cd ~/nccl-tests/
make MPI=1
```

Test on all three nodes (replace `spark1`, `spark2`, and `spark3` with the actual hostnames or IP addresses on the non-QSFP interface):

```bash
# Set environment variables
export CUDA_HOME="/usr/local/cuda"
export MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
export NCCL_HOME="$HOME/nccl/build/"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH"

# For 3-node mesh we have to use 10G interface for OOB communication!
export UCX_NET_DEVICES=enP7s7
export NCCL_SOCKET_IFNAME=enP7s7
export OMPI_MCA_btl_tcp_if_include=enP7s7
export NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1
export NCCL_IB_DISABLE=0

# Run the all_gather performance test across all three nodes
mpirun -np 3 -H spark1:1,spark2:1,spark3:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -x NCCL_IB_MERGE_NICS=0 -x NCCL_NET_PLUGIN=none -x NCCL_IB_SUBNET_AWARE_ROUTING=1 \
  $HOME/nccl-tests/build/all_gather_perf -b 16G -e 16G -f 3
```

## Closed rings of four or more Sparks

A closed ring connects each Spark's two QSFP ports to two different neighbors.
Keep all nodes on the management network and configure passwordless SSH similarly to other topologies. 

Assign each link's addressed CX-7 interfaces their
own subnet, as in the three-node mesh setup above. Discovery requires Python 3,
`ibdev2netdev`, `ip`, and `ping` on every node.

### Example: four-node ring

Use four QSFP cables to connect `spark1 -> spark2 -> spark3 -> spark4 -> spark1`.
Connect port 0 of each Spark to port 1 of the next Spark, including the closing
cable from `spark4` to `spark1`:

```mermaid
flowchart LR
    s1["spark1 (head)"]
    s2["spark2"]
    s3["spark3"]
    s4["spark4"]
    s1 <-->|"spark1 port 0 / spark2 port 1"| s2
    s2 <-->|"spark2 port 0 / spark3 port 1"| s3
    s3 <-->|"spark3 port 0 / spark4 port 1"| s4
    s4 <-->|"spark4 port 0 / spark1 port 1"| s1
```

All four Sparks also connect to the same management LAN through their RJ-45
ports. The example assumes these management identities on `enP7s7`:

| Node | Management address | Rank |
| :--- | :--- | :--- |
| `spark1` | `192.0.2.11` | 0 (head) |
| `spark2` | `192.0.2.12` | 1 |
| `spark3` | `192.0.2.13` | 2 |
| `spark4` | `192.0.2.14` | 3 |

These are documentation addresses; use the nodes' actual management addresses.
Keep the management LAN configuration in its existing netplan file. The CX-7
examples below use eight separate `/24` subnets: two per cable, one for each
PCIe rail. Choose subnets that do not overlap your other networks.

| Cable | First endpoint | Second endpoint | CX-7 subnets |
| :--- | :--- | :--- | :--- |
| 1 | `spark1` port 0 | `spark2` port 1 | `10.20.0.0/24`, `10.20.1.0/24` |
| 2 | `spark2` port 0 | `spark3` port 1 | `10.20.2.0/24`, `10.20.3.0/24` |
| 3 | `spark3` port 0 | `spark4` port 1 | `10.20.4.0/24`, `10.20.5.0/24` |
| 4 | `spark4` port 0 | `spark1` port 1 | `10.20.6.0/24`, `10.20.7.0/24` |

Port 0 uses `enp1s0f0np0` and `enP2p1s0f0np0`; port 1 uses
`enp1s0f1np1` and `enP2p1s0f1np1`. Both endpoints of each rail must use the
same subnet and MTU. This example uses MTU 9000 throughout.

Create `/etc/netplan/40-cx7.yaml` on each node with its corresponding content
below. If those interfaces already have netplan definitions, update those
definitions to match the example instead of adding duplicate entries.

**spark1:**

```yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [10.20.0.11/24]
    enP2p1s0f0np0:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [10.20.1.11/24]
    enp1s0f1np1:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [10.20.6.11/24]
    enP2p1s0f1np1:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [10.20.7.11/24]
```

**spark2:**

```yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [10.20.2.12/24]
    enP2p1s0f0np0:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [10.20.3.12/24]
    enp1s0f1np1:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [10.20.0.12/24]
    enP2p1s0f1np1:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [10.20.1.12/24]
```

**spark3:**

```yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [10.20.4.13/24]
    enP2p1s0f0np0:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [10.20.5.13/24]
    enp1s0f1np1:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [10.20.2.13/24]
    enP2p1s0f1np1:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [10.20.3.13/24]
```

**spark4:**

```yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [10.20.6.14/24]
    enP2p1s0f0np0:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [10.20.7.14/24]
    enp1s0f1np1:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [10.20.4.14/24]
    enP2p1s0f1np1:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: 9000
      addresses: [10.20.5.14/24]
```

On each node, validate and apply its configuration:

```bash
sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan generate
sudo netplan apply
ibdev2netdev
```

All four RoCE interfaces should report `Up`. Check the two neighbors from each
node, binding each ping to the matching interface. For example, on `spark1`:

```bash
# spark2, reached through spark1 port 0
ping -c 3 -I enp1s0f0np0 10.20.0.12
ping -c 3 -I enP2p1s0f0np0 10.20.1.12
# spark4, reached through spark1 port 1
ping -c 3 -I enp1s0f1np1 10.20.6.14
ping -c 3 -I enP2p1s0f1np1 10.20.7.14
```

Use the existing [passwordless SSH configurator](#passwordless-ssh-and-benchmarks)
to enable access from the head to every node. With `spark1` as head, the expected
rank order is `spark1, spark2, spark3, spark4`; `spark3` is the indirect copy
destination and is reached through `spark2`. Discovery derives that path from
the links, without extra SSH-user or jump-host fields in `.env`.

### Discovery, distribution, and launch

Management discovery already finds the whole cluster. With four or more nodes
in mesh mode, discovery also inspects the active RoCE interfaces on each node
and verifies matching subnets in both directions. It counts the twin rails as
one neighbor. When the resulting graph is a closed ring, it orders
`CLUSTER_NODES` along that ring, with the local node first, and saves the link
endpoints in the same `.env` file:

```bash
./run-recipe.sh --discover
```

The only new field is `CLUSTER_LINKS`: a single-line JSON list of links, enclosed
in single quotes. Each link maps two management IPv4 addresses to their
respective CX-7 IPv4 addresses. For the four-node example above, the generated
rank order and first rail of each cable look like this:

```dotenv
CLUSTER_NODES=192.0.2.11,192.0.2.12,192.0.2.13,192.0.2.14
CLUSTER_LINKS='[{"192.0.2.11":"10.20.0.11","192.0.2.12":"10.20.0.12"},{"192.0.2.12":"10.20.2.12","192.0.2.13":"10.20.2.13"},{"192.0.2.13":"10.20.4.13","192.0.2.14":"10.20.4.14"},{"192.0.2.14":"10.20.6.14","192.0.2.11":"10.20.6.11"}]'
```

This example shows one rail per cable; discovery saves every verified addressed
rail. Existing `LOCAL_IP`, `ETH_IF`, and `IB_IF` fields remain in the file.
Discovery also saves the existing NCCL settings `CONTAINER_NCCL_ALGO=Ring`,
`CONTAINER_NCCL_NET_PLUGIN=none`, `CONTAINER_NCCL_IB_SUBNET_AWARE_ROUTING=1`, and
`CONTAINER_NCCL_IB_MERGE_NICS=0`. There is no additional topology type, rank-order
list, SSH user, or saved copy-route table.

For a ring, launch validation checks the selected ranks, including any trimming
caused by the requested parallelism. The native backend preserves rank placement;
Ray and mixed TP/PP/DP layouts are currently rejected for a sparse ring. Use the
full ring with tensor parallelism, a directly connected pair, or solo mode.
Selecting three nodes from a four-node ring breaks its closing link and is
rejected before containers start. Ring defaults are supplied when omitted;
conflicting NCCL settings are rejected. Automatic NCCL algorithm selection may
attempt connections between non-neighbor nodes, which caused QP timeouts in the
four-node validation. SSH jump paths do not provide RDMA routing for NCCL.

Image and model distribution derive shortest paths through the saved links.
Direct neighbors receive transfers directly; other nodes use SSH `ProxyJump`
through as many intermediate nodes as needed, without staging weights or images
on those nodes. These are TCP transfers over CX-7; every jump host must permit
SSH TCP forwarding.
Existing SSH identities, authentication, and host-key configuration are reused.
Temporary SSH routing files are removed when each transfer finishes.

There is no four-node limit in discovery or distribution. For example, the
farthest destination in a six-node ring uses two jump hosts; in an eight-node
ring it uses three. Mocked integration tests cover discovery, rank validation,
and image/model distribution to every worker in both serial and parallel modes
for those sizes. Physical NCCL inference has been validated on four nodes;
larger-ring inference still needs validation with the intended model and tensor
parallel size. Each copy originates at the head, so concurrent transfers share
link bandwidth and SSH forwarding capacity; distribution throughput does not
scale linearly with the number of nodes.

```bash
# Preview the four-node launch before setup
./run-recipe.sh recipes/qwen3.8-27b-nvfp4-dflash2.yaml --tp 4 --dry-run
# Download/build as needed, distribute to all workers, and launch
./run-recipe.sh recipes/qwen3.8-27b-nvfp4-dflash2.yaml --tp 4 --setup
```

To distribute an existing image or a model separately:

```bash
./build-and-copy.sh --no-build -c --copy-parallel
./hf-download.sh org/model -c --copy-parallel
```

New ring configs omit `COPY_HOSTS`, so `-c` includes every worker. Explicit
`--copy-to` hosts or an explicitly saved `COPY_HOSTS` subset still select only
those destinations. Management addresses and saved link addresses are accepted
as destinations. Recipe setup forwards the selected config to both copy scripts,
compares image IDs through the same routes, and distributes models even when
already cached on the head.

Two-node, three-node mesh, and switch-connected configurations retain their
existing discovery, NCCL, and `COPY_HOSTS` behavior. A complete graph discovered
with four active NICs follows the legacy copy scan. Autodiscovery leaves
`NCCL_ALGO` unset on those topologies; it sets `Ring` for a true ring. Existing
config files without `CLUSTER_LINKS` continue to work unchanged; rediscover to
enable ring support.
Saved links describe the network at discovery time. Run discovery again after
recabling or changing addresses; copy failures do not silently fall back to the
management network.
