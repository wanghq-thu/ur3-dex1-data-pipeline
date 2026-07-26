"""
通过 RTDE 读取并保存 UR3 机械臂的六个关节角。

依赖安装：
    pip install ur_rtde

运行方式：
    python3 read_ur3_rtde.py
    python3 read_ur3_rtde.py <robot_ip>
    python3 read_ur3_rtde.py --output <output.jsonl>

默认连接地址为 192.168.1.10。程序读取并打印 10 次关节角，
读取频率为 125 Hz（周期 8 ms），关节角单位为弧度（rad）。
时间戳使用主机单调时钟，单位为微秒（us）。

每个样本以一行 JSON 写入 ur3_joint_states.jsonl，字段包括：
    sample_index：样本序号
    host_timestamp_us：主机单调时钟
    robot_timestamp_s：UR 控制器启动后的时间
    actual_q_rad：六个实际关节角

注意：actual_q_rad 是机械臂状态（observation），不是控制指令
（action）。用于模仿学习时还需要单独记录实际发送给机械臂的指令。
"""

import argparse
import json
import time
from pathlib import Path

import rtde_receive

READ_FREQUENCY_HZ = 125.0
SAMPLE_COUNT = 10
DEFAULT_OUTPUT_PATH = Path("ur3_joint_states.jsonl")


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
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "JSONL 输出路径"
            "（默认：ur3_joint_states.jsonl）"
        ),
    )
    args = parser.parse_args()

    receiver = rtde_receive.RTDEReceiveInterface(
        args.robot_ip,
        READ_FREQUENCY_HZ,
    )

    try:
        with args.output.open(
            "w",
            encoding="utf-8",
            buffering=1,
        ) as output_file:
            for index in range(SAMPLE_COUNT):
                period_start = receiver.initPeriod()

                joint_angles = receiver.getActualQ()
                robot_timestamp_s = receiver.getTimestamp()
                host_timestamp_us = (
                    time.monotonic_ns() // 1000
                )

                sample = {
                    "sample_index": index,
                    "host_timestamp_us": (
                        host_timestamp_us
                    ),
                    "robot_timestamp_s": (
                        robot_timestamp_s
                    ),
                    "actual_q_rad": list(joint_angles),
                }
                sample_json = json.dumps(
                    sample,
                    separators=(",", ":"),
                )

                output_file.write(sample_json + "\n")
                print(sample_json)

                receiver.waitPeriod(period_start)
    finally:
        receiver.disconnect()


if __name__ == "__main__":
    main()
