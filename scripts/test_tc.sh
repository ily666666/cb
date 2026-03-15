#!/bin/bash
# ============================================================
# tc 限速测试脚本（两台 Ubuntu VM 间验证）
# ============================================================
# 用法: sudo ./test_tc.sh <网卡名> <iperf3服务端IP>
# 示例: sudo ./test_tc.sh ens33 192.168.1.102
#
# 前提: 对方机器已运行 iperf3 -s
# ============================================================

set -e

IFACE=${1:-ens33}
SERVER_IP=${2:-192.168.1.102}

if [ "$EUID" -ne 0 ]; then
    echo "请使用 sudo 运行此脚本"
    exit 1
fi

echo "============================================"
echo " tc 限速验证测试"
echo " 本机网卡: $IFACE"
echo " 目标服务端: $SERVER_IP"
echo "============================================"
echo ""
echo "请确保对方机器已运行: iperf3 -s"
echo "按回车开始..."
read

# 加载内核模块（防止报错）
modprobe sch_htb 2>/dev/null || true

# ------ 测试 1: 不限速（基线）------
echo ""
echo "====== 测试 1/3: 不限速（基线）======"
tc qdisc del dev $IFACE root 2>/dev/null || true
sleep 1
iperf3 -c $SERVER_IP -t 5 2>&1 | tail -3
BASELINE=$(iperf3 -c $SERVER_IP -t 3 2>&1 | grep sender | awk '{print $7, $8}')
echo "基线带宽: $BASELINE"
echo ""

# ------ 测试 2: 限速 80Mbps (10 MB/s) ------
echo "====== 测试 2/3: 限速 80Mbps (= 10 MB/s) ======"
tc qdisc del dev $IFACE root 2>/dev/null || true
tc qdisc add dev $IFACE root handle 1: htb default 10
tc class add dev $IFACE parent 1: classid 1:10 htb rate 80mbit ceil 80mbit
sleep 1
iperf3 -c $SERVER_IP -t 5 2>&1 | tail -3
echo ""

# ------ 测试 3: 限速 800Mbps (100 MB/s) ------
echo "====== 测试 3/3: 限速 800Mbps (= 100 MB/s) ======"
tc qdisc del dev $IFACE root 2>/dev/null || true
tc qdisc add dev $IFACE root handle 1: htb default 10
tc class add dev $IFACE parent 1: classid 1:10 htb rate 800mbit ceil 800mbit
sleep 1
iperf3 -c $SERVER_IP -t 5 2>&1 | tail -3
echo ""

# ------ 清除 ------
tc qdisc del dev $IFACE root 2>/dev/null || true

echo "============================================"
echo " 测试完成！"
echo ""
echo " 预期结果:"
echo "   测试1 不限速:  ~940 Mbps"
echo "   测试2 限80M:   ~80 Mbps  (10 MB/s)"
echo "   测试3 限800M:  ~800 Mbps (100 MB/s)"
echo ""
echo " 如果三次差异明显 → tc 限速生效"
echo " 已自动清除所有限速规则"
echo "============================================"
