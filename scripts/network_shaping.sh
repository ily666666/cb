#!/bin/bash
# ============================================================
# 云边端协同系统 — 单网卡网络限速脚本（tc 方案）
# ============================================================
# 固定限速：端↔边 100 MB/s (800Mbps)，边↔云 8 MB/s (64Mbps)
#
# 用法: sudo ./network_shaping.sh <角色> [网卡名]
#   角色: cloud / edge / device / clear
#   网卡名: 默认 eth0
#
# 示例:
#   sudo ./network_shaping.sh cloud
#   sudo ./network_shaping.sh edge eth0
#   sudo ./network_shaping.sh device ens33
#   sudo ./network_shaping.sh clear
# ============================================================

set -e

ROLE=$1
IFACE=${2:-eth0}

# ====== IP 配置（按实际环境修改）======
CLOUD_IP=192.168.1.10
EDGE1_IP=192.168.1.21
EDGE2_IP=192.168.1.22
DEVICE1_IP=192.168.1.31
DEVICE2_IP=192.168.1.32
DEVICE3_IP=192.168.1.33
DEVICE4_IP=192.168.1.34

# ====== 固定带宽 ======
DE_RATE="800mbit"   # 端↔边：100 MB/s = 800 Mbps
EC_RATE="64mbit"    # 边↔云：8 MB/s   = 64 Mbps

if [ -z "$ROLE" ]; then
    echo "用法: sudo $0 <角色> [网卡名]"
    echo "  角色: cloud / edge / device / clear"
    echo ""
    echo "固定限速: 端↔边 100 MB/s (800Mbps), 边↔云 8 MB/s (64Mbps)"
    echo ""
    echo "示例:"
    echo "  sudo $0 cloud"
    echo "  sudo $0 edge eth0"
    echo "  sudo $0 device ens33"
    echo "  sudo $0 clear"
    exit 1
fi

# 清除旧规则
tc qdisc del dev $IFACE root 2>/dev/null || true

if [ "$ROLE" = "clear" ]; then
    echo "[限速] 已清除 $IFACE 上的所有规则"
    exit 0
fi

echo "============================================================"
echo " 固定限速: 端↔边 100 MB/s (800Mbps), 边↔云 8 MB/s (64Mbps)"
echo " 网卡: $IFACE"
echo "============================================================"

# 创建根队列，默认流量不限速
tc qdisc add dev $IFACE root handle 1: htb default 99
tc class add dev $IFACE parent 1: classid 1:99 htb rate 1000mbit

if [ "$ROLE" = "device" ]; then
    # === 端→边：800Mbps (100 MB/s) ===
    tc class add dev $IFACE parent 1: classid 1:10 htb rate $DE_RATE ceil $DE_RATE
    tc filter add dev $IFACE parent 1: protocol ip prio 1 u32 match ip dst $EDGE1_IP/32 flowid 1:10
    tc filter add dev $IFACE parent 1: protocol ip prio 1 u32 match ip dst $EDGE2_IP/32 flowid 1:10

    # === 端→云：64Mbps (8 MB/s) ===
    tc class add dev $IFACE parent 1: classid 1:20 htb rate $EC_RATE ceil $EC_RATE
    tc filter add dev $IFACE parent 1: protocol ip prio 1 u32 match ip dst $CLOUD_IP/32 flowid 1:20

    echo "[端节点] 限速已设置:"
    echo "  → 边侧 ($EDGE1_IP, $EDGE2_IP): 100 MB/s (800Mbps)"
    echo "  → 云侧 ($CLOUD_IP): 8 MB/s (64Mbps)"

elif [ "$ROLE" = "edge" ]; then
    # === 边→云：64Mbps (8 MB/s) ===
    tc class add dev $IFACE parent 1: classid 1:10 htb rate $EC_RATE ceil $EC_RATE
    tc filter add dev $IFACE parent 1: protocol ip prio 1 u32 match ip dst $CLOUD_IP/32 flowid 1:10

    # === 边→端：800Mbps (100 MB/s) ===
    tc class add dev $IFACE parent 1: classid 1:20 htb rate $DE_RATE ceil $DE_RATE
    tc filter add dev $IFACE parent 1: protocol ip prio 1 u32 match ip dst $DEVICE1_IP/32 flowid 1:20
    tc filter add dev $IFACE parent 1: protocol ip prio 1 u32 match ip dst $DEVICE2_IP/32 flowid 1:20
    tc filter add dev $IFACE parent 1: protocol ip prio 1 u32 match ip dst $DEVICE3_IP/32 flowid 1:20
    tc filter add dev $IFACE parent 1: protocol ip prio 1 u32 match ip dst $DEVICE4_IP/32 flowid 1:20

    echo "[边节点] 限速已设置:"
    echo "  → 云侧 ($CLOUD_IP): 8 MB/s (64Mbps)"
    echo "  → 端侧: 100 MB/s (800Mbps)"

elif [ "$ROLE" = "cloud" ]; then
    # === 云→边：64Mbps (8 MB/s) ===
    tc class add dev $IFACE parent 1: classid 1:10 htb rate $EC_RATE ceil $EC_RATE
    tc filter add dev $IFACE parent 1: protocol ip prio 1 u32 match ip dst $EDGE1_IP/32 flowid 1:10
    tc filter add dev $IFACE parent 1: protocol ip prio 1 u32 match ip dst $EDGE2_IP/32 flowid 1:10

    echo "[云节点] 限速已设置:"
    echo "  → 边侧 ($EDGE1_IP, $EDGE2_IP): 8 MB/s (64Mbps)"
else
    echo "错误: 未知角色 '$ROLE'，可选 cloud / edge / device / clear"
    exit 1
fi

echo ""
echo "[当前规则]"
tc class show dev $IFACE
echo ""
echo "验证: iperf3 -c <目标IP> -t 10"
echo "清除: sudo $0 clear $IFACE"
