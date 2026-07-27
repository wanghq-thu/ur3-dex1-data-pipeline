"""
通过键盘控制录制 ROS 2 中的 UR3 关节状态。

运行：
    source /opt/ros/humble/setup.bash
    source /home/wanghq/git_rep/wanghq-thu/gello_software/.venv-ros2/bin/activate
    python record_ur3_joint_states_ros2.py

键盘控制：
    S：开始一个新的 episode
    E：结束当前 episode
    Q：结束当前 episode 并退出
    Ctrl+C：结束当前 episode 并退出

默认订阅：
    /ur3/joint_states    sensor_msgs/msg/JointState

默认输出目录：
    recordings/ur3_joint_states

可以使用 ROS 参数修改输出目录：
    python record_ur3_joint_states_ros2.py --ros-args \
        -p output_dir:=/path/to/output

每个 episode 保存为独立的 JSONL 文件。时间戳单位为微秒，
关节角单位为弧度。
"""

from __future__ import annotations

import json
import os
import select
import sys
import termios
import threading
import tty
from datetime import datetime
from pathlib import Path
from typing import TextIO

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

JOINT_STATE_TOPIC = "/ur3/joint_states"
DEFAULT_OUTPUT_DIR = Path("recordings/ur3_joint_states")


def joint_state_to_record(
    message: JointState,
    episode_index: int,
    sample_index: int,
) -> dict[str, object]:
    """将 JointState 转换为一个可写入 JSONL 的样本。"""

    timestamp_us = (
        message.header.stamp.sec * 1_000_000
        + message.header.stamp.nanosec // 1000
    )

    return {
        "episode_index": episode_index,
        "sample_index": sample_index,
        "timestamp_us": timestamp_us,
        "frame_id": message.header.frame_id,
        "joint_names": list(message.name),
        "joint_angles_rad": list(message.position),
    }


class UR3JointStateRecorder(Node):
    """订阅 UR3 关节状态，并按 episode 写入 JSONL。"""

    def __init__(self) -> None:
        super().__init__("ur3_joint_state_recorder")

        self.declare_parameter(
            "output_dir",
            str(DEFAULT_OUTPUT_DIR),
        )
        self._output_dir = Path(
            str(self.get_parameter("output_dir").value)
        )
        self._output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._recording_lock = threading.Lock()
        self._output_file: TextIO | None = None
        self._output_path: Path | None = None
        self._episode_index = 0
        self._sample_index = 0

        self._subscription = self.create_subscription(
            JointState,
            JOINT_STATE_TOPIC,
            self._joint_state_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"Listening on {JOINT_STATE_TOPIC}"
        )
        self.get_logger().info(
            "Keyboard: S=start, E=end, Q=quit"
        )

    def start_recording(self) -> bool:
        """开始一个新 episode；已经在录制时返回 False。"""

        with self._recording_lock:
            if self._output_file is not None:
                self.get_logger().warning(
                    "Already recording; press E before S"
                )
                return False

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )
            filename = (
                f"ur3_joint_states_{timestamp}"
                f"_episode_{self._episode_index:03d}.jsonl"
            )
            output_path = self._output_dir / filename
            output_file = output_path.open(
                "w",
                encoding="utf-8",
                buffering=1,
            )

            self._output_file = output_file
            self._output_path = output_path
            self._sample_index = 0

            episode_index = self._episode_index
            self._episode_index += 1

        self.get_logger().info(
            f"Recording episode {episode_index}: "
            f"{output_path}"
        )
        return True

    def stop_recording(
        self,
        *,
        warn_if_inactive: bool = True,
    ) -> bool:
        """结束当前 episode；没有在录制时返回 False。"""

        with self._recording_lock:
            if self._output_file is None:
                if warn_if_inactive:
                    self.get_logger().warning(
                        "Not recording; press S to start"
                    )
                return False

            output_file = self._output_file
            output_path = self._output_path
            sample_count = self._sample_index
            episode_index = self._episode_index - 1

            output_file.flush()
            output_file.close()

            self._output_file = None
            self._output_path = None
            self._sample_index = 0

        self.get_logger().info(
            f"Stopped episode {episode_index}: "
            f"{sample_count} samples, {output_path}"
        )
        return True

    def _joint_state_callback(
        self,
        message: JointState,
    ) -> None:
        with self._recording_lock:
            if self._output_file is None:
                return

            record = joint_state_to_record(
                message,
                self._episode_index - 1,
                self._sample_index,
            )
            self._output_file.write(
                json.dumps(
                    record,
                    separators=(",", ":"),
                )
                + "\n"
            )
            self._sample_index += 1

    def close(self) -> None:
        """安全结束可能仍在进行的录制。"""

        self.stop_recording(
            warn_if_inactive=False,
        )


def keyboard_loop(
    node: UR3JointStateRecorder,
    stop_event: threading.Event,
) -> None:
    """非阻塞监听 S/E/Q，并保证退出时恢复终端设置。"""

    file_descriptor = sys.stdin.fileno()
    original_settings = termios.tcgetattr(
        file_descriptor
    )

    try:
        tty.setraw(file_descriptor)

        while not stop_event.is_set():
            readable, _, _ = select.select(
                [file_descriptor],
                [],
                [],
                0.1,
            )

            if not readable:
                continue

            key = os.read(
                file_descriptor,
                1,
            ).decode(
                errors="ignore",
            ).lower()

            if key == "s":
                node.start_recording()
            elif key == "e":
                node.stop_recording()
            elif key in ("q", "\x03"):
                node.stop_recording(
                    warn_if_inactive=False,
                )
                stop_event.set()
    except Exception as exc:
        node.get_logger().error(
            f"Keyboard listener failed: {exc}"
        )
        stop_event.set()
    finally:
        termios.tcsetattr(
            file_descriptor,
            termios.TCSADRAIN,
            original_settings,
        )


def main(args: list[str] | None = None) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError(
            "需要在交互式终端中运行，才能使用 S/E/Q 控制录制"
        )

    rclpy.init(args=args)
    node: UR3JointStateRecorder | None = None
    keyboard_thread: threading.Thread | None = None
    stop_event = threading.Event()

    try:
        node = UR3JointStateRecorder()
        keyboard_thread = threading.Thread(
            target=keyboard_loop,
            args=(node, stop_event),
            name="keyboard-listener",
            daemon=True,
        )
        keyboard_thread.start()

        while (
            rclpy.ok()
            and not stop_event.is_set()
        ):
            rclpy.spin_once(
                node,
                timeout_sec=0.1,
            )
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()

        if node is not None:
            node.close()

        if keyboard_thread is not None:
            keyboard_thread.join(timeout=1.0)

        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
