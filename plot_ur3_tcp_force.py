"""
实时绘制并保存 UR3 机械臂的三方向末端力。

依赖安装：
    pip install ur_rtde "matplotlib>=3.9,<3.10"

运行方式：
    python3 plot_ur3_tcp_force.py
    python3 plot_ur3_tcp_force.py <robot_ip>
    python3 plot_ur3_tcp_force.py --output <output.jsonl>

默认连接地址为 192.168.1.10。程序通过后台线程以 125 Hz
读取末端力，并以约 25 Hz 刷新最近 10 秒的 Fx、Fy、Fz
折线图。关闭绘图窗口或按 Ctrl+C 可停止程序。

所有样本会写入 ur3_tcp_forces_plot.jsonl。每行字段包括：
    sample_index：样本序号
    host_timestamp_us：主机单调时钟，单位为微秒（us）
    robot_timestamp_s：UR 控制器启动后的时间，单位为秒（s）
    actual_tcp_force_n_nm：[Fx, Fy, Fz, Tx, Ty, Tz]

Fx、Fy、Fz 的单位为牛顿（N），Tx、Ty、Tz 的单位为牛米
（N·m）。数据来自控制器的 actual_TCP_force，方向与机器人
基坐标系一致，作用点为工具法兰；程序不进行滤波、零点校正
或坐标转换。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import matplotlib

# 当前 ROS 2 虚拟环境同时包含 Qt 和系统 Matplotlib。Qt 在缺少
# libxcb-cursor 时会直接终止进程，因此默认使用系统已有的 Tk。
# 仍可通过 MPLBACKEND 显式选择其他后端（例如测试时使用 Agg）。
if "MPLBACKEND" not in os.environ:
    matplotlib.use("TkAgg")

# 系统 Matplotlib 的 mpl_toolkits 会导致无关的 3D 导入警告；
# 本程序仅使用二维坐标轴，可安全忽略这一条特定警告。
warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
    module=r"matplotlib\.projections",
)

import matplotlib.pyplot as plt
import rtde_receive
from matplotlib.animation import FuncAnimation

READ_FREQUENCY_HZ = 125.0
PLOT_UPDATE_INTERVAL_MS = 40
HISTORY_SECONDS = 10.0
HISTORY_SAMPLE_COUNT = int(
    READ_FREQUENCY_HZ * HISTORY_SECONDS
)
DEFAULT_ROBOT_IP = "192.168.1.10"
DEFAULT_OUTPUT_PATH = Path(
    "ur3_tcp_forces_plot.jsonl"
)
WRENCH_ELEMENT_COUNT = 6

PlotSample = tuple[float, float, float, float]


@dataclass
class AcquisitionState:
    """采集线程向绘图线程报告的状态。"""

    error: Exception | None = None


def put_latest_sample(
    plot_queue: queue.Queue[PlotSample],
    sample: PlotSample,
) -> None:
    """将最新数据放入有界队列，队列满时丢弃最旧绘图点。"""

    try:
        plot_queue.put_nowait(sample)
        return
    except queue.Full:
        pass

    try:
        plot_queue.get_nowait()
    except queue.Empty:
        pass

    try:
        plot_queue.put_nowait(sample)
    except queue.Full:
        # 消费线程可能恰好改变了队列；文件记录不受影响。
        pass


def acquire_tcp_forces(
    receiver: rtde_receive.RTDEReceiveInterface,
    output_file: TextIO,
    plot_queue: queue.Queue[PlotSample],
    stop_event: threading.Event,
    state: AcquisitionState,
) -> None:
    """持续采集六维末端力，保存数据并发送绘图点。"""

    sample_index = 0
    first_host_timestamp_us: int | None = None

    try:
        while not stop_event.is_set():
            period_start = receiver.initPeriod()

            tcp_force = list(
                receiver.getActualTCPForce()
            )
            if len(tcp_force) != WRENCH_ELEMENT_COUNT:
                raise RuntimeError(
                    "actual_TCP_force 应包含 6 个元素，"
                    f"实际收到 {len(tcp_force)} 个"
                )
            if not all(
                math.isfinite(value)
                for value in tcp_force
            ):
                raise RuntimeError(
                    "actual_TCP_force 包含非有限数值"
                )

            robot_timestamp_s = receiver.getTimestamp()
            host_timestamp_us = (
                time.monotonic_ns() // 1000
            )
            if first_host_timestamp_us is None:
                first_host_timestamp_us = (
                    host_timestamp_us
                )

            sample = {
                "sample_index": sample_index,
                "host_timestamp_us": (
                    host_timestamp_us
                ),
                "robot_timestamp_s": (
                    robot_timestamp_s
                ),
                "actual_tcp_force_n_nm": tcp_force,
            }
            output_file.write(
                json.dumps(
                    sample,
                    separators=(",", ":"),
                )
                + "\n"
            )

            elapsed_s = (
                host_timestamp_us
                - first_host_timestamp_us
            ) / 1_000_000.0
            put_latest_sample(
                plot_queue,
                (
                    elapsed_s,
                    tcp_force[0],
                    tcp_force[1],
                    tcp_force[2],
                ),
            )

            sample_index += 1
            receiver.waitPeriod(period_start)
    except Exception as exc:
        if not stop_event.is_set():
            state.error = exc
    finally:
        stop_event.set()


class TCPForcePlotter:
    """绘制最近一段时间内的 Fx、Fy、Fz。"""

    def __init__(
        self,
        plot_queue: queue.Queue[PlotSample],
        stop_event: threading.Event,
        state: AcquisitionState,
    ) -> None:
        self._plot_queue = plot_queue
        self._stop_event = stop_event
        self._state = state

        self._times: deque[float] = deque(
            maxlen=HISTORY_SAMPLE_COUNT
        )
        self._force_x: deque[float] = deque(
            maxlen=HISTORY_SAMPLE_COUNT
        )
        self._force_y: deque[float] = deque(
            maxlen=HISTORY_SAMPLE_COUNT
        )
        self._force_z: deque[float] = deque(
            maxlen=HISTORY_SAMPLE_COUNT
        )

        self.figure, self._axes = plt.subplots()
        (self._line_x,) = self._axes.plot(
            [],
            [],
            label="Fx",
        )
        (self._line_y,) = self._axes.plot(
            [],
            [],
            label="Fy",
        )
        (self._line_z,) = self._axes.plot(
            [],
            [],
            label="Fz",
        )
        self._lines = (
            self._line_x,
            self._line_y,
            self._line_z,
        )

        self._axes.set_title(
            "UR3 TCP Force (last 10 seconds)"
        )
        self._axes.set_xlabel("Elapsed time (s)")
        self._axes.set_ylabel("Force (N)")
        self._axes.set_xlim(0.0, HISTORY_SECONDS)
        self._axes.grid(True, alpha=0.3)
        self._axes.legend(loc="upper right")
        self.figure.tight_layout()
        self.figure.canvas.mpl_connect(
            "close_event",
            self._on_close,
        )

        self._animation = FuncAnimation(
            self.figure,
            self._update,
            interval=PLOT_UPDATE_INTERVAL_MS,
            blit=False,
            cache_frame_data=False,
        )

    def _on_close(self, _event: object) -> None:
        self._stop_event.set()

    def _update(
        self,
        _frame: object,
    ) -> tuple[object, object, object]:
        while True:
            try:
                elapsed_s, force_x, force_y, force_z = (
                    self._plot_queue.get_nowait()
                )
            except queue.Empty:
                break

            self._times.append(elapsed_s)
            self._force_x.append(force_x)
            self._force_y.append(force_y)
            self._force_z.append(force_z)

        self._line_x.set_data(
            self._times,
            self._force_x,
        )
        self._line_y.set_data(
            self._times,
            self._force_y,
        )
        self._line_z.set_data(
            self._times,
            self._force_z,
        )

        if self._times:
            latest_time = self._times[-1]
            window_start = max(
                0.0,
                latest_time - HISTORY_SECONDS,
            )
            window_end = max(
                HISTORY_SECONDS,
                latest_time,
            )
            self._axes.set_xlim(
                window_start,
                window_end,
            )
            self._update_y_limits()

        if self._state.error is not None:
            plt.close(self.figure)

        return self._lines

    def _update_y_limits(self) -> None:
        visible_forces = (
            list(self._force_x)
            + list(self._force_y)
            + list(self._force_z)
        )
        minimum = min(visible_forces)
        maximum = max(visible_forces)
        padding = max(
            (maximum - minimum) * 0.1,
            1.0,
        )
        self._axes.set_ylim(
            minimum - padding,
            maximum + padding,
        )

    def show(self) -> None:
        plt.show()

    def close(self) -> None:
        plt.close(self.figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实时绘制并保存 UR3 三方向末端力"
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


def run(
    robot_ip: str,
    output_path: Path,
) -> None:
    receiver = rtde_receive.RTDEReceiveInterface(
        robot_ip,
        READ_FREQUENCY_HZ,
    )
    stop_event = threading.Event()
    state = AcquisitionState()
    plot_queue: queue.Queue[PlotSample] = queue.Queue(
        maxsize=HISTORY_SAMPLE_COUNT
    )
    plotter: TCPForcePlotter | None = None

    try:
        plotter = TCPForcePlotter(
            plot_queue,
            stop_event,
            state,
        )
        with output_path.open(
            "w",
            encoding="utf-8",
            buffering=1,
        ) as output_file:
            worker = threading.Thread(
                target=acquire_tcp_forces,
                args=(
                    receiver,
                    output_file,
                    plot_queue,
                    stop_event,
                    state,
                ),
                name="ur3-tcp-force-reader",
            )
            worker.start()

            try:
                plotter.show()
            finally:
                stop_event.set()
                worker.join()
    finally:
        if plotter is not None:
            plotter.close()
        receiver.disconnect()

    if state.error is not None:
        raise RuntimeError(
            f"RTDE 末端力采集失败：{state.error}"
        ) from state.error


def main() -> None:
    args = parse_args()

    try:
        run(args.robot_ip, args.output)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
