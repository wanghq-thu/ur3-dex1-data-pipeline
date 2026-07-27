"""
通过 RTDE 持续读取并保存 UR3 机械臂的末端力/力矩。

依赖安装：
    pip install ur_rtde

运行方式：
    python3 read_ur3_tcp_force.py
    python3 read_ur3_tcp_force.py <robot_ip>
    python3 read_ur3_tcp_force.py --output <output.jsonl>

默认连接地址为 192.168.1.10。程序以 125 Hz（周期 8 ms）
持续读取、打印并保存末端力，按 Ctrl+C 停止。时间戳使用主机
单调时钟，单位为微秒（us）。

终端使用紧凑格式显示样本编号、三轴力和三轴力矩，例如：
    #12 F[N]=(-3.28,-3.89,51.40) T[Nm]=(0.55,-3.95,-0.74)

每个样本以一行 JSON 写入 ur3_tcp_forces.jsonl，字段包括：
    sample_index：样本序号
    host_timestamp_us：主机单调时钟
    robot_timestamp_s：UR 控制器启动后的时间
    actual_tcp_force_n_nm：[Fx, Fy, Fz, Tx, Ty, Tz]

Fx、Fy、Fz 的单位为牛顿（N），Tx、Ty、Tz 的单位为牛米
（N·m）。该数据来自控制器的 actual_TCP_force，方向与机器人
基坐标系一致，作用点为工具法兰；程序不进行滤波或软件清零。
实际测量方式和精度取决于机器人型号、控制器版本以及正确的
负载和 TCP 配置。
"""

import argparse
import json
import time
from pathlib import Path
from typing import TextIO

import rtde_receive

READ_FREQUENCY_HZ = 125.0
DEFAULT_ROBOT_IP = "192.168.1.10"
DEFAULT_OUTPUT_PATH = Path("ur3_tcp_forces.jsonl")
WRENCH_ELEMENT_COUNT = 6


def format_console_output(
    sample_index: int,
    tcp_force: list[float],
) -> str:
    """生成适合单行显示的紧凑末端力信息。"""

    force_values = ",".join(
        f"{value:.2f}" for value in tcp_force[:3]
    )
    torque_values = ",".join(
        f"{value:.2f}" for value in tcp_force[3:]
    )

    return (
        f"#{sample_index} "
        f"F[N]=({force_values}) "
        f"T[Nm]=({torque_values})"
    )


def read_tcp_forces(
    receiver: rtde_receive.RTDEReceiveInterface,
    output_file: TextIO,
) -> None:
    """持续读取末端力，将每个样本打印并写入 JSONL。"""

    sample_index = 0

    while True:
        period_start = receiver.initPeriod()

        tcp_force = list(receiver.getActualTCPForce())
        if len(tcp_force) != WRENCH_ELEMENT_COUNT:
            raise RuntimeError(
                "actual_TCP_force 应包含 6 个元素，"
                f"实际收到 {len(tcp_force)} 个"
            )

        robot_timestamp_s = receiver.getTimestamp()
        host_timestamp_us = time.monotonic_ns() // 1000

        sample = {
            "sample_index": sample_index,
            "host_timestamp_us": host_timestamp_us,
            "robot_timestamp_s": robot_timestamp_s,
            "actual_tcp_force_n_nm": tcp_force,
        }
        sample_json = json.dumps(
            sample,
            separators=(",", ":"),
        )

        output_file.write(sample_json + "\n")
        print(
            format_console_output(
                sample_index,
                tcp_force,
            ),
            flush=True,
        )

        sample_index += 1
        receiver.waitPeriod(period_start)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="读取 UR3 的末端力和力矩"
    )
    parser.add_argument(
        "robot_ip",
        nargs="?",
        default=DEFAULT_ROBOT_IP,
        help=(
            "UR3 控制器的 IP 地址"
            f"（默认：{DEFAULT_ROBOT_IP}）"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "JSONL 输出路径"
            f"（默认：{DEFAULT_OUTPUT_PATH}）"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
            read_tcp_forces(receiver, output_file)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.disconnect()


if __name__ == "__main__":
    main()
