"""
通过 RTDE 读取 UR3 机械臂的六个关节角。

依赖安装：
    pip install ur_rtde

运行方式：
    python3 read_ur3_rtde.py
    python3 read_ur3_rtde.py <robot_ip>

默认连接地址为 192.168.1.10。程序读取并打印 10 次关节角，
相邻两次读取间隔 0.1 秒，关节角单位为弧度（rad）。
"""

import argparse
import time

import rtde_receive


def main() -> None:
    parser = argparse.ArgumentParser(
        description="读取 UR3 的六个关节角"
    )
    parser.add_argument(
        "robot_ip",  # 没有--，是位置参数
        nargs="?",  # ?意思是参数是可选的，最多可以有一个值
        default="192.168.1.10",
        help="UR3 控制器的 IP 地址（默认：192.168.1.10）",
    )
    args = parser.parse_args()

    receiver = rtde_receive.RTDEReceiveInterface(
        args.robot_ip
    )

    try:
        for index in range(10):
            joint_angles = receiver.getActualQ()
            print(f"{index + 1}: {joint_angles} rad")
            time.sleep(0.1)
    finally:
        receiver.disconnect()


if __name__ == "__main__":
    main()
