"""
订阅 ROS 2 中的 UR3 关节状态并实时输出。

运行：
    source /opt/ros/humble/setup.bash
    source /home/wanghq/git_rep/wanghq-thu/gello_software/.venv-ros2/bin/activate
    python read_ur3_joint_states_ros2.py

默认订阅：
    /ur3/joint_states    sensor_msgs/msg/JointState

每收到一条消息，就以一行 JSON 输出样本序号、ROS 时间戳、
关节名称和关节角。时间戳单位为微秒，关节角单位为弧度。
"""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

JOINT_STATE_TOPIC = "/ur3/joint_states"


def joint_state_to_record(
    message: JointState,
    sample_index: int,
) -> dict[str, object]:
    """将 JointState 转换为便于处理的字典。"""

    timestamp_us = (
        message.header.stamp.sec * 1_000_000
        + message.header.stamp.nanosec // 1000
    )

    return {
        "sample_index": sample_index,
        "timestamp_us": timestamp_us,
        "joint_names": list(message.name),
        "joint_angles_rad": list(message.position),
    }


class UR3JointStateReader(Node):
    """订阅并输出 UR3 关节状态。"""

    def __init__(self) -> None:
        super().__init__("ur3_joint_state_reader")

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

    def _joint_state_callback(
        self,
        message: JointState,
    ) -> None:
        record = joint_state_to_record(
            message,
            self._sample_index,
        )
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
    node = UR3JointStateReader()

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
