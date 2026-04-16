#!/bin/bash
# ============================================================
# 网络拥塞模拟 — 用 iperf3 占用 2MB/s 带宽
# ============================================================
#
# 需要两台机器配合：
#   接收端(靶机): 随便一台能通的机器，负责接收垃圾流量
#   发送端(本机): 想制造拥塞的机器，出口带宽会被占用
#
# === 接收端(靶机) 运行 ===
#   ./network_congestion.sh server          # 启动接收服务(后台)
#   ./network_congestion.sh server-stop     # 停止接收服务
#
# === 发送端(本机) 运行 ===
#   ./network_congestion.sh start --to <靶机IP>                # 自动选路
#   ./network_congestion.sh start --from <本机IP> --to <靶机IP> # 指定出口
#   ./network_congestion.sh stop                                # 停止占用
#
# === 查看状态 ===
#   ./network_congestion.sh status          # 查看运行状态
#
# 示例:
#   靶机(192.168.1.20):  ./network_congestion.sh server
#   本机:                 ./network_congestion.sh start --to 192.168.1.20
#   本机(指定出口):       ./network_congestion.sh start --from 192.168.1.100 --to 192.168.1.20
#   本机:                 ./network_congestion.sh stop
#   靶机:                 ./network_congestion.sh server-stop
# ============================================================

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
LOG_DIR="${SCRIPT_DIR}/congestion_logs"
mkdir -p "$LOG_DIR"

if ! command -v iperf3 &> /dev/null; then
    echo "[提示] 未检测到 iperf3，正在自动安装..."
    sudo apt install iperf3 -y
fi

case "${1}" in

server)
    pkill -f "iperf3 -s" 2>/dev/null || true
    iperf3 -s -D --logfile "$LOG_DIR/server.log"
    echo "[接收端] iperf3 服务端已启动(后台运行，端口5201)"
    echo "日志: $LOG_DIR/server.log"
    echo "停止: $0 server-stop"
    ;;

server-stop)
    pkill -f "iperf3 -s" 2>/dev/null && echo "[接收端] 已停止 iperf3 服务端" || echo "没有运行中的 iperf3 服务端"
    ;;

start)
    shift
    FROM_IP=""
    TO_IP=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --from) FROM_IP="$2"; shift 2 ;;
            --to)   TO_IP="$2";   shift 2 ;;
            *)      echo "未知参数: $1"; exit 1 ;;
        esac
    done

    if [ -z "$TO_IP" ]; then
        echo "用法: $0 start --to <靶机IP> [--from <本机IP>]"
        echo ""
        echo "示例:"
        echo "  $0 start --to 192.168.1.20                          # 自动选路"
        echo "  $0 start --from 192.168.1.100 --to 192.168.1.20     # 指定出口"
        exit 1
    fi

    pkill -f "iperf3 -c" 2>/dev/null || true
    BIND_OPT=""
    if [ -n "$FROM_IP" ]; then
        BIND_OPT="--bind $FROM_IP"
    fi
    nohup iperf3 -c $TO_IP $BIND_OPT -b 16M -t 0 > "$LOG_DIR/client.log" 2>&1 &

    if [ -n "$FROM_IP" ]; then
        echo "[发送端] $FROM_IP --> $TO_IP  占用 2MB/s(后台运行)"
    else
        echo "[发送端] 本机 --> $TO_IP  占用 2MB/s(后台运行，自动选路)"
    fi
    echo "日志: $LOG_DIR/client.log"
    echo "停止: $0 stop"
    ;;

stop)
    pkill -f "iperf3 -c" 2>/dev/null && echo "[发送端] 已停止背景流量" || echo "没有运行中的背景流量"
    ;;

status)
    echo "=== iperf3 进程状态 ==="
    ps aux | grep "iperf3" | grep -v grep || echo "无 iperf3 进程"
    ;;

*)
    echo "用法:"
    echo "  接收端(靶机):  $0 server                                    # 启动接收服务"
    echo "                 $0 server-stop                               # 停止接收服务"
    echo "  发送端(本机):  $0 start --to <靶机IP>                       # 自动选路，占用 2MB/s"
    echo "                 $0 start --from <本机IP> --to <靶机IP>       # 指定出口，占用 2MB/s"
    echo "                 $0 stop                                      # 停止占用"
    echo "  查看状态:      $0 status"
    exit 1
    ;;

esac
