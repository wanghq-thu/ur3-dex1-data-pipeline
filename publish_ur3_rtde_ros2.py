"""
通过 RTDE 读取 UR3 状态，并实时发布为 ROS 2 消息。

环境：
    Ubuntu 22.04
    ROS 2 Humble

运行：
    source /opt/ros/humble/setup.bash
    source /home/wanghq/git_rep/wanghq-thu/gello_software/.venv-ros2/bin/activate
    python publish_ur3_rtde_ros2.py

也可以不激活虚拟环境，直接指定解释器：
    source /opt/ros/humble/setup.bash
    /home/wanghq/git_rep/wanghq-thu/gello_software/.venv-ros2/bin/python \
        publish_ur3_rtde_ros2.py

可选参数：
    python publish_ur3_rtde_ros2.py --ros-args \
        -p robot_ip:=192.168.1.10 \
        -p frequency:=125.0 \
        -p base_frame:=base

发布话题：
    /ur3/joint_states    sensor_msgs/msg/JointState
    /ur3/tcp_pose        geometry_msgs/msg/PoseStamped

末端位姿中的 [rx, ry, rz] 是 UR 使用的旋转向量。本节点会将其
转换为 PoseStamped 所需的四元数。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import rclpy
import rtde_receive
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

DEFAULT_ROBOT_IP = "192.168.1.10"
DEFAULT_FREQUENCY_HZ = 125.0
DEFAULT_BASE_FRAME = "base"

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def rotation_vector_to_quaternion(
    rotation_vector: Sequence[float],
) -> tuple[float, float, float, float]:
    """将轴角旋转向量转换为四元数 (x, y, z, w)。"""

    rx, ry, rz = rotation_vector
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)

    if angle < 1e-12:
        return 0.0, 0.0, 0.0, 1.0

    scale = math.sin(angle / 2.0) / angle

    return (
        rx * scale,
        ry * scale,
        rz * scale,
        math.cos(angle / 2.0),
    )


class UR3RTDEPublisher(Node):
    """以固定频率发布 UR3 关节角和末端位姿。"""

    def __init__(self) -> None:
        super().__init__("ur3_rtde_publisher")

        self.declare_parameter(
            "robot_ip",
            DEFAULT_ROBOT_IP,
        )
        self.declare_parameter(
            "frequency",
            DEFAULT_FREQUENCY_HZ,
        )
        self.declare_parameter(
            "base_frame",
            DEFAULT_BASE_FRAME,
        )

        robot_ip = str(
            self.get_parameter("robot_ip").value
        )
        frequency_hz = float(
            self.get_parameter("frequency").value
        )
        self._base_frame = str(
            self.get_parameter("base_frame").value
        )

        if frequency_hz <= 0.0:
            raise ValueError("frequency 必须大于 0")

        self._receiver = (
            rtde_receive.RTDEReceiveInterface(
                robot_ip,
                frequency_hz,
            )
        )

        self._joint_publisher = self.create_publisher(
            JointState,
            "/ur3/joint_states",
            qos_profile_sensor_data,
        )
        self._tcp_pose_publisher = self.create_publisher(
            PoseStamped,
            "/ur3/tcp_pose",
            qos_profile_sensor_data,
        )

        self._timer = self.create_timer(
            1.0 / frequency_hz,
            self._publish_state,
        )

        self.get_logger().info(
            f"Connected to UR3 at {robot_ip}; "
            f"publishing at {frequency_hz:.1f} Hz"
        )

    def _publish_state(self) -> None:
        joint_angles = self._receiver.getActualQ()
        tcp_pose = self._receiver.getActualTCPPose()
        stamp = self.get_clock().now().to_msg()

        joint_message = JointState()
        joint_message.header.stamp = stamp
        joint_message.header.frame_id = self._base_frame
        joint_message.name = JOINT_NAMES
        joint_message.position = list(joint_angles)
        self._joint_publisher.publish(joint_message)

        quaternion = rotation_vector_to_quaternion(
            tcp_pose[3:6]
        )

        pose_message = PoseStamped()
        pose_message.header.stamp = stamp
        pose_message.header.frame_id = self._base_frame
        pose_message.pose.position.x = tcp_pose[0]
        pose_message.pose.position.y = tcp_pose[1]
        pose_message.pose.position.z = tcp_pose[2]
        pose_message.pose.orientation.x = quaternion[0]
        pose_message.pose.orientation.y = quaternion[1]
        pose_message.pose.orientation.z = quaternion[2]
        pose_message.pose.orientation.w = quaternion[3]
        self._tcp_pose_publisher.publish(pose_message)

    def disconnect(self) -> None:
        """断开 RTDE 连接。"""

        self._receiver.disconnect()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: UR3RTDEPublisher | None = None

    try:
        node = UR3RTDEPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.disconnect()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
