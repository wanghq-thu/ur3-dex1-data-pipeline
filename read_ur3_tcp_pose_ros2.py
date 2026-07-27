"""
订阅 ROS 2 中的 UR3 末端位姿并实时输出。

运行：
    source /opt/ros/humble/setup.bash
    source /home/wanghq/git_rep/wanghq-thu/gello_software/.venv-ros2/bin/activate
    python read_ur3_tcp_pose_ros2.py

默认订阅：
    /ur3/tcp_pose    geometry_msgs/msg/PoseStamped

每收到一条消息，就以一行 JSON 输出样本序号、ROS 时间戳、
参考坐标系、末端位置、四元数和 UR 常用旋转向量。
时间戳单位为微秒，位置单位为米，旋转向量单位为弧度。
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

TCP_POSE_TOPIC = "/ur3/tcp_pose"
QUATERNION_EPSILON = 1e-12


def quaternion_to_rotation_vector(
    quaternion_xyzw: Sequence[float],
) -> list[float]:
    """将四元数 (x, y, z, w) 转换为旋转向量。"""

    x, y, z, w = quaternion_xyzw
    norm = math.sqrt(
        x * x + y * y + z * z + w * w
    )

    if norm < QUATERNION_EPSILON:
        raise ValueError("收到的末端姿态是零四元数")

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    # q 和 -q 表示同一个旋转。固定符号后，转换结果保持一致，
    # 并优先得到旋转角不超过 pi 的旋转向量。
    for component in (w, x, y, z):
        if abs(component) > QUATERNION_EPSILON:
            if component < 0.0:
                x, y, z, w = -x, -y, -z, -w
            break

    vector_norm = math.sqrt(x * x + y * y + z * z)

    if vector_norm < QUATERNION_EPSILON:
        return [0.0, 0.0, 0.0]

    angle = 2.0 * math.atan2(vector_norm, w)
    scale = angle / vector_norm

    return [
        x * scale,
        y * scale,
        z * scale,
    ]


def pose_stamped_to_record(
    message: PoseStamped,
    sample_index: int,
) -> dict[str, object]:
    """将 PoseStamped 转换为便于处理的字典。"""

    timestamp_us = (
        message.header.stamp.sec * 1_000_000
        + message.header.stamp.nanosec // 1000
    )
    quaternion_xyzw = [
        message.pose.orientation.x,
        message.pose.orientation.y,
        message.pose.orientation.z,
        message.pose.orientation.w,
    ]

    return {
        "sample_index": sample_index,
        "timestamp_us": timestamp_us,
        "frame_id": message.header.frame_id,
        "tcp_position_m": [
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
        ],
        "tcp_orientation_quaternion_xyzw": (
            quaternion_xyzw
        ),
        "tcp_rotation_vector_rad": (
            quaternion_to_rotation_vector(
                quaternion_xyzw
            )
        ),
    }


class UR3TCPPoseReader(Node):
    """订阅并输出 UR3 末端位姿。"""

    def __init__(self) -> None:
        super().__init__("ur3_tcp_pose_reader")

        self._sample_index = 0
        self._subscription = self.create_subscription(
            PoseStamped,
            TCP_POSE_TOPIC,
            self._tcp_pose_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"Listening on {TCP_POSE_TOPIC}"
        )

    def _tcp_pose_callback(
        self,
        message: PoseStamped,
    ) -> None:
        try:
            record = pose_stamped_to_record(
                message,
                self._sample_index,
            )
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return

        self._sample_index += 1

        print(
            json.dumps(
                record,
                separators=(",", ":"),
            ),
            flush=True,
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = UR3TCPPoseReader()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
