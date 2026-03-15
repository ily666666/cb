# tc 限速测试指南（两台 Ubuntu 虚拟机）

## 一、环境准备

### 1.1 VM 网络配置

两台 Ubuntu 虚拟机需要能互相 ping 通，推荐使用 **桥接模式** 或 **Host-Only 网络**：

| 虚拟机 | 角色 | IP（示例） |
|--------|------|----------|
| VM-A | 发送方（模拟边侧） | 192.168.1.101 |
| VM-B | 接收方（模拟云侧） | 192.168.1.102 |

**确认网络连通：**

```bash
# 在 VM-A 上
ping 192.168.1.102
```

**查看网卡名：**

```bash
ip addr
# 找到有 IP 地址的网卡，通常是 ens33、eth0、enp0s3 等
```

### 1.2 安装测试工具

两台机器都执行：

```bash
sudo apt update
sudo apt install -y iperf3
```

---

## 二、测试步骤

### 第一步：不限速，测原始带宽

先看一下不限速时两台 VM 之间的实际带宽（作为基线）。

**VM-B（接收方）启动 iperf3 服务端：**

```bash
iperf3 -s
```

**VM-A（发送方）测试带宽：**

```bash
iperf3 -c 192.168.1.102 -t 10
```

预期输出（示例）：

```
[ ID] Interval           Transfer     Bitrate
[  5]   0.00-10.00  sec  1.10 GBytes   941 Mbits/sec    sender
```

> 记下这个基线值（通常接近千兆 ~940Mbps）。

---

### 第二步：在 VM-A 上设置 tc 限速到 80Mbps (10MB/s)

```bash
# 替换 ens33 为你实际的网卡名
IFACE=ens33

# 清除旧规则
sudo tc qdisc del dev $IFACE root 2>/dev/null

# 设置限速 80Mbps（= 10 MB/s）
sudo tc qdisc add dev $IFACE root handle 1: htb default 10
sudo tc class add dev $IFACE parent 1: classid 1:10 htb rate 80mbit ceil 80mbit
```

**确认规则已生效：**

```bash
tc qdisc show dev $IFACE
tc class show dev $IFACE
```

应看到类似输出：

```
qdisc htb 1: root ... default 0x10
class htb 1:10 root rate 80Mbit ceil 80Mbit ...
```

---

### 第三步：再次测速，验证限速效果

**VM-B 上 iperf3 服务端保持运行（或重新启动）：**

```bash
iperf3 -s
```

**VM-A 发起测试：**

```bash
iperf3 -c 192.168.1.102 -t 10
```

预期输出：

```
[ ID] Interval           Transfer     Bitrate
[  5]   0.00-10.00  sec  95.2 MBytes  79.8 Mbits/sec    sender
```

> 看到 Bitrate 从 ~940Mbps 降到 **~80Mbps**，说明限速生效。

---

### 第四步：换一个速率试试（800Mbps = 100MB/s）

```bash
IFACE=ens33

# 清除旧规则
sudo tc qdisc del dev $IFACE root 2>/dev/null

# 设置限速 800Mbps（= 100 MB/s）
sudo tc qdisc add dev $IFACE root handle 1: htb default 10
sudo tc class add dev $IFACE parent 1: classid 1:10 htb rate 800mbit ceil 800mbit
```

**再测：**

```bash
iperf3 -c 192.168.1.102 -t 10
```

预期看到 ~800Mbps（VM 虚拟网卡可能跑不满，接近即可）。

---

### 第五步：测完清除限速

```bash
sudo tc qdisc del dev ens33 root
```

验证已清除：

```bash
tc qdisc show dev ens33
# 应该只剩默认的 pfifo_fast 或 fq_codel
```

---

## 三、完整测试脚本（复制即用）

在 VM-A 上保存为 `test_tc.sh`，一键完成全部测试：

```bash
#!/bin/bash
# tc 限速测试脚本
# 用法: sudo ./test_tc.sh <网卡名> <iperf3服务端IP>
# 示例: sudo ./test_tc.sh ens33 192.168.1.102

IFACE=${1:-ens33}
SERVER_IP=${2:-192.168.1.102}

echo "============================================"
echo " tc 限速测试"
echo " 网卡: $IFACE"
echo " 目标: $SERVER_IP"
echo "============================================"
echo ""
echo "请确保对方机器已运行: iperf3 -s"
echo "按回车开始测试..."
read

# --- 测试 1: 不限速 ---
echo ""
echo "====== 测试 1: 不限速（基线）======"
tc qdisc del dev $IFACE root 2>/dev/null
iperf3 -c $SERVER_IP -t 5
echo ""

# --- 测试 2: 限速 80Mbps (10 MB/s) ---
echo "====== 测试 2: 限速 80Mbps (10 MB/s) ======"
tc qdisc del dev $IFACE root 2>/dev/null
tc qdisc add dev $IFACE root handle 1: htb default 10
tc class add dev $IFACE parent 1: classid 1:10 htb rate 80mbit ceil 80mbit
echo "规则:"
tc class show dev $IFACE
echo ""
iperf3 -c $SERVER_IP -t 5
echo ""

# --- 测试 3: 限速 800Mbps (100 MB/s) ---
echo "====== 测试 3: 限速 800Mbps (100 MB/s) ======"
tc qdisc del dev $IFACE root 2>/dev/null
tc qdisc add dev $IFACE root handle 1: htb default 10
tc class add dev $IFACE parent 1: classid 1:10 htb rate 800mbit ceil 800mbit
echo "规则:"
tc class show dev $IFACE
echo ""
iperf3 -c $SERVER_IP -t 5
echo ""

# --- 清除 ---
echo "====== 清除所有限速 ======"
tc qdisc del dev $IFACE root 2>/dev/null
echo "已清除"

echo ""
echo "============================================"
echo " 测试完成！对比三次结果即可确认 tc 是否生效"
echo "============================================"
```

---

## 四、预期结果汇总

| 测试 | tc 设置 | 预期 iperf3 结果 |
|------|--------|-----------------|
| 测试 1 | 不限速 | ~940 Mbps |
| 测试 2 | 80 Mbps | **~80 Mbps (10 MB/s)** |
| 测试 3 | 800 Mbps | **~800 Mbps (100 MB/s)** |

如果三次结果呈现明显的阶梯差异，说明 tc 限速完全生效。

---

## 五、常见问题

### Q: tc 命令报错 "RTNETLINK answers: No such file or directory"

```bash
# 加载内核模块
sudo modprobe sch_htb
```

### Q: 限速不生效，iperf3 速率没变化

1. 确认网卡名正确：`ip addr` 查看
2. 确认规则存在：`tc qdisc show dev ens33`
3. tc 只限制**出方向**，如果你在接收方设的 tc，对发送方无效。**限速要设在发送方**

### Q: VMware/VirtualBox 虚拟网卡速率本身就很低

虚拟机网络性能受宿主机和虚拟化方式影响，基线可能只有 200-500Mbps。只要限速到 80Mbps 后明显低于基线，就说明 tc 生效。
