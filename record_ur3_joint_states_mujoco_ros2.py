"""
录制 ROS 2 中的 UR3 关节角，并使用 MuJoCo UR5e 模型实时显示。

UR3 和 UR5e 的六个关节具有相同名称和拓扑结构，因此这里直接将
UR3 关节角写入 UR5e 模型。UR5e 只用于近似可视化，录制的数据仍是
ROS 话题中的原始 UR3 关节状态。

运行：
    source /opt/ros/humble/setup.bash
    source /home/wanghq/git_rep/wanghq-thu/gello_software/.venv-ros2/bin/activate
    python record_ur3_joint_states_mujoco_ros2.py

键盘控制（终端或 MuJoCo 窗口均可）：
    S：开始一个新的 episode
    E：结束当前 episode
    Q：结束当前 episode 并退出
    Ctrl+C：结束当前 episode 并退出

默认订阅：
    /ur3/joint_states    sensor_msgs/msg/JointState

可用 ROS 参数：
    output_dir          JSONL 输出目录
    mujoco_model_path   MuJoCo UR5e scene.xml 路径
    viewer_hz           MuJoCo 显示刷新率，默认 60 Hz

示例：
    python record_ur3_joint_states_mujoco_ros2.py --ros-args \
        -p output_dir:=recordings/ur3_joint_states \
        -p viewer_hz:=60.0

ROS 数据接收和录制频率由发布端决定（当前为 125 Hz）；MuJoCo 只在
每次刷新时显示最新关节状态，不会降低录制频率。
"""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

import mujoco
import mujoco.viewer
import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState

from record_ur3_joint_states_ros2 import (
    UR3JointStateRecorder,
    keyboard_loop,
)

UR_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

DEFAULT_MUJOCO_MODEL_PATH = Path(
    "/home/wanghq/git_rep/wanghq-thu/gello_software/"
    "third_party/mujoco_menagerie/"
    "universal_robots_ur5e/scene.xml"
)
DEFAULT_VIEWER_HZ = 60.0
WARNING_INTERVAL_S = 2.0


def ordered_joint_angles(
    joint_names: Sequence[str],
    joint_positions: Sequence[float],
) -> tuple[float, ...]:
    """按 UR 关节顺序提取有限的关节角，不依赖消息数组顺序。"""

    if len(joint_names) != len(joint_positions):
        raise ValueError(
            "joint name and position counts differ: "
            f"{len(joint_names)} != {len(joint_positions)}"
        )

    if len(set(joint_names)) != len(joint_names):
        raise ValueError("joint names contain duplicates")

    positions_by_name = dict(
        zip(joint_names, joint_positions)
    )
    missing_names = [
        name
        for name in UR_JOINT_NAMES
        if name not in positions_by_name
    ]
    if missing_names:
        raise ValueError(
            "missing required joints: "
            + ", ".join(missing_names)
        )

    angles = tuple(
        float(positions_by_name[name])
        for name in UR_JOINT_NAMES
    )
    if not all(math.isfinite(angle) for angle in angles):
        raise ValueError("joint positions contain non-finite values")

    return angles


def mujoco_joint_qpos_addresses(
    model: mujoco.MjModel,
) -> tuple[int, ...]:
    """查找 UR 六个关节在 MuJoCo qpos 中的地址。"""

    addresses: list[int] = []
    for name in UR_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        if joint_id < 0:
            raise ValueError(
                f"MuJoCo model is missing joint: {name}"
            )

        joint_type = model.jnt_type[joint_id]
        if joint_type not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            raise ValueError(
                f"MuJoCo joint has unsupported type: {name}"
            )

        addresses.append(
            int(model.jnt_qposadr[joint_id])
        )

    return tuple(addresses)


class UR3JointStateMujocoRecorder(
    UR3JointStateRecorder
):
    """在原有 JSONL 录制功能上缓存最新的可视化关节角。"""

    def __init__(self) -> None:
        self._joint_state_lock = threading.Lock()
        self._latest_joint_angles: tuple[
            float, ...
        ] | None = None
        self._last_warning_time = 0.0

        super().__init__()

        self.declare_parameter(
            "mujoco_model_path",
            str(DEFAULT_MUJOCO_MODEL_PATH),
        )
        self.declare_parameter(
            "viewer_hz",
            DEFAULT_VIEWER_HZ,
        )

        self.mujoco_model_path = Path(
            str(
                self.get_parameter(
                    "mujoco_model_path"
                ).value
            )
        ).expanduser()
        self.viewer_hz = float(
            self.get_parameter("viewer_hz").value
        )
        if not math.isfinite(self.viewer_hz):
            raise ValueError(
                "viewer_hz must be a finite number"
            )
        if self.viewer_hz <= 0.0:
            raise ValueError(
                "viewer_hz must be greater than zero"
            )

    def _joint_state_callback(
        self,
        message: JointState,
    ) -> None:
        """缓存可视化状态，并始终保留原有录制行为。"""

        try:
            angles = ordered_joint_angles(
                message.name,
                message.position,
            )
        except ValueError as exc:
            self._warn_invalid_joint_state(str(exc))
        else:
            with self._joint_state_lock:
                self._latest_joint_angles = angles

        super()._joint_state_callback(message)

    def latest_joint_angles(
        self,
    ) -> tuple[float, ...] | None:
        """返回最新关节角的线程安全副本。"""

        with self._joint_state_lock:
            return self._latest_joint_angles

    def _warn_invalid_joint_state(
        self,
        reason: str,
    ) -> None:
        now = time.monotonic()
        if (
            now - self._last_warning_time
            < WARNING_INTERVAL_S
        ):
            return

        self._last_warning_time = now
        self.get_logger().warning(
            "Skipping MuJoCo update: " + reason
        )


def handle_viewer_key(
    keycode: int,
    node: UR3JointStateMujocoRecorder,
    stop_event: threading.Event,
) -> None:
    """处理 MuJoCo 窗口中的 S/E/Q 按键。"""

    if keycode in (ord("S"), ord("s")):
        node.start_recording()
    elif keycode in (ord("E"), ord("e")):
        node.stop_recording()
    elif keycode in (ord("Q"), ord("q")):
        node.stop_recording(
            warn_if_inactive=False,
        )
        stop_event.set()


def spin_ros(
    executor: SingleThreadedExecutor,
    node: UR3JointStateMujocoRecorder,
    stop_event: threading.Event,
) -> None:
    """在后台运行 ROS executor，并将异常传递为停止请求。"""

    try:
        executor.spin()
    except Exception as exc:
        if rclpy.ok():
            node.get_logger().error(
                f"ROS executor failed: {exc}"
            )
        stop_event.set()


def run_viewer(
    node: UR3JointStateMujocoRecorder,
    stop_event: threading.Event,
    executor: SingleThreadedExecutor,
) -> None:
    """启动后台输入/ROS线程，并在主线程运行 MuJoCo 窗口。"""

    model_path = node.mujoco_model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            "MuJoCo model not found: "
            f"{model_path}. Use --ros-args "
            "-p mujoco_model_path:=/path/to/scene.xml"
        )

    model = mujoco.MjModel.from_xml_path(
        str(model_path)
    )
    joint_addresses = mujoco_joint_qpos_addresses(
        model
    )
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(
            model,
            data,
            0,
        )
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    key_callback = lambda keycode: handle_viewer_key(
        keycode,
        node,
        stop_event,
    )

    ros_thread: threading.Thread | None = None
    keyboard_thread: threading.Thread | None = None

    try:
        with mujoco.viewer.launch_passive(
            model,
            data,
            key_callback=key_callback,
        ) as viewer:
            ros_thread = threading.Thread(
                target=spin_ros,
                args=(executor, node, stop_event),
                name="ros-executor",
                daemon=True,
            )
            keyboard_thread = threading.Thread(
                target=keyboard_loop,
                args=(node, stop_event),
                name="keyboard-listener",
                daemon=True,
            )

            ros_thread.start()
            keyboard_thread.start()

            node.get_logger().info(
                f"MuJoCo model: {model_path}"
            )
            node.get_logger().info(
                f"Viewer refresh rate: {node.viewer_hz:g} Hz"
            )
            node.get_logger().info(
                "Viewer keyboard: S=start, E=end, Q=quit"
            )

            frame_period = 1.0 / node.viewer_hz
            next_frame_time = time.monotonic()
            displayed_angles: tuple[
                float, ...
            ] | None = None

            while (
                rclpy.ok()
                and not stop_event.is_set()
                and viewer.is_running()
            ):
                latest_angles = (
                    node.latest_joint_angles()
                )
                if (
                    latest_angles is not None
                    and latest_angles
                    != displayed_angles
                ):
                    with viewer.lock():
                        for address, angle in zip(
                            joint_addresses,
                            latest_angles,
                        ):
                            data.qpos[address] = angle
                        mujoco.mj_forward(
                            model,
                            data,
                        )
                    displayed_angles = latest_angles

                viewer.sync()

                next_frame_time += frame_period
                wait_time = (
                    next_frame_time
                    - time.monotonic()
                )
                if wait_time > 0.0:
                    stop_event.wait(wait_time)
                else:
                    next_frame_time = (
                        time.monotonic()
                    )
    finally:
        stop_event.set()

        if keyboard_thread is not None:
            keyboard_thread.join(timeout=1.0)

        executor.shutdown(timeout_sec=1.0)

        if ros_thread is not None:
            ros_thread.join(timeout=1.0)


def main(args: list[str] | None = None) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError(
            "需要在交互式终端中运行，才能使用 S/E/Q 控制录制"
        )

    rclpy.init(args=args)
    node: UR3JointStateMujocoRecorder | None = None
    executor: SingleThreadedExecutor | None = None
    stop_event = threading.Event()

    try:
        node = UR3JointStateMujocoRecorder()
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        run_viewer(
            node,
            stop_event,
            executor,
        )
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()

        if executor is not None:
            executor.shutdown(timeout_sec=1.0)

        if node is not None:
            node.close()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
