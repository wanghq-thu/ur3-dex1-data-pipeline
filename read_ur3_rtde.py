import argparse
import time

import rtde_receive


def main() -> None:
    parser = argparse.ArgumentParser(
        description="读取 UR3 的六个关节角"
    )
    parser.add_argument(
        "robot_ip",
        nargs="?",
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
