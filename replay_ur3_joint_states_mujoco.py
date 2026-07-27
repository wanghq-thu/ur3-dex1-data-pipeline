"""
在 MuJoCo 中回放录制的 UR3 关节角 JSONL 文件。

程序使用 MuJoCo Menagerie 的 UR5e 模型近似显示 UR3。回放时间由
每条记录的 timestamp_us 决定，因此不会依赖固定的采样频率。

运行：
    source /home/wanghq/git_rep/wanghq-thu/gello_software/.venv-ros2/bin/activate
    python replay_ur3_joint_states_mujoco.py \
        recordings/ur3_joint_states/ur3_joint_states_xxx.jsonl

MuJoCo 窗口键盘控制：
    Space：暂停或继续
    R：从头重新播放
    Q：退出

可选参数：
    --speed 0.5       以 0.5 倍速回放
    --speed 2.0       以 2 倍速回放
    --loop            循环回放
    --viewer-hz 60    设置显示刷新率
    --model PATH      指定其他 MuJoCo scene.xml
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import mujoco.viewer

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


@dataclass(frozen=True)
class JointSample:
    """一条用于回放的关节状态。"""

    timestamp_us: int
    joint_angles_rad: tuple[float, ...]


def ordered_joint_angles(
    joint_names: Sequence[str],
    joint_positions: Sequence[float],
) -> tuple[float, ...]:
    """按 UR 关节顺序提取关节角，不依赖 JSON 数组顺序。"""

    if len(joint_names) != len(joint_positions):
        raise ValueError(
            "joint name and angle counts differ: "
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
        raise ValueError("joint angles contain non-finite values")

    return angles


def load_joint_samples(
    recording_path: Path,
) -> list[JointSample]:
    """读取并验证关节状态 JSONL 文件。"""

    if not recording_path.is_file():
        raise FileNotFoundError(
            f"recording file not found: {recording_path}"
        )

    samples: list[JointSample] = []
    previous_timestamp_us: int | None = None

    with recording_path.open(
        "r",
        encoding="utf-8",
    ) as recording_file:
        for line_number, line in enumerate(
            recording_file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                timestamp_us = record["timestamp_us"]
                if (
                    not isinstance(timestamp_us, int)
                    or isinstance(timestamp_us, bool)
                ):
                    raise ValueError(
                        "timestamp_us must be an integer"
                    )

                joint_names = record["joint_names"]
                joint_angles = record[
                    "joint_angles_rad"
                ]
                if not isinstance(joint_names, list):
                    raise ValueError(
                        "joint_names must be a list"
                    )
                if not isinstance(joint_angles, list):
                    raise ValueError(
                        "joint_angles_rad must be a list"
                    )

                ordered_angles = ordered_joint_angles(
                    joint_names,
                    joint_angles,
                )
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    f"{recording_path}:{line_number}: "
                    f"invalid record: {exc}"
                ) from exc

            if (
                previous_timestamp_us is not None
                and timestamp_us
                < previous_timestamp_us
            ):
                raise ValueError(
                    f"{recording_path}:{line_number}: "
                    "timestamp_us is not monotonic"
                )

            samples.append(
                JointSample(
                    timestamp_us=timestamp_us,
                    joint_angles_rad=ordered_angles,
                )
            )
            previous_timestamp_us = timestamp_us

    if not samples:
        raise ValueError(
            f"recording file contains no samples: "
            f"{recording_path}"
        )

    return samples


def mujoco_joint_qpos_addresses(
    model: mujoco.MjModel,
) -> tuple[int, ...]:
    """查找六个 UR 关节在 MuJoCo qpos 中的地址。"""

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


def set_mujoco_joint_angles(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_addresses: Sequence[int],
    joint_angles: Sequence[float],
) -> None:
    """直接设置模型关节角并更新正向运动学。"""

    for address, angle in zip(
        joint_addresses,
        joint_angles,
    ):
        data.qpos[address] = angle
    mujoco.mj_forward(model, data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a recorded UR3 joint-state JSONL "
            "file with a MuJoCo UR5e model."
        )
    )
    parser.add_argument(
        "recording",
        type=Path,
        help="recorded joint-state JSONL file",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MUJOCO_MODEL_PATH,
        help=(
            "MuJoCo scene.xml path "
            f"(default: {DEFAULT_MUJOCO_MODEL_PATH})"
        ),
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--viewer-hz",
        type=float,
        default=DEFAULT_VIEWER_HZ,
        help=(
            "viewer refresh rate in Hz "
            f"(default: {DEFAULT_VIEWER_HZ:g})"
        ),
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="restart automatically after reaching the end",
    )
    args = parser.parse_args()

    if not math.isfinite(args.speed) or args.speed <= 0.0:
        parser.error("--speed must be a finite number greater than zero")
    if (
        not math.isfinite(args.viewer_hz)
        or args.viewer_hz <= 0.0
    ):
        parser.error(
            "--viewer-hz must be a finite number "
            "greater than zero"
        )

    return args


def main() -> None:
    args = parse_args()

    recording_path = args.recording.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"MuJoCo model not found: {model_path}"
        )

    samples = load_joint_samples(recording_path)
    relative_timestamps_us = [
        sample.timestamp_us - samples[0].timestamp_us
        for sample in samples
    ]
    duration_us = relative_timestamps_us[-1]

    model = mujoco.MjModel.from_xml_path(
        str(model_path)
    )
    data = mujoco.MjData(model)
    joint_addresses = mujoco_joint_qpos_addresses(
        model
    )
    set_mujoco_joint_angles(
        model,
        data,
        joint_addresses,
        samples[0].joint_angles_rad,
    )

    commands: queue.SimpleQueue[str] = (
        queue.SimpleQueue()
    )

    def key_callback(keycode: int) -> None:
        if keycode == ord(" "):
            commands.put("toggle_pause")
        elif keycode in (ord("R"), ord("r")):
            commands.put("restart")
        elif keycode in (ord("Q"), ord("q")):
            commands.put("quit")

    sample_periods = len(samples) - 1
    duration_s = duration_us / 1_000_000
    average_hz = (
        sample_periods / duration_s
        if duration_s > 0.0
        else 0.0
    )

    print(f"Recording: {recording_path}")
    print(f"Samples: {len(samples)}")
    print(f"Duration: {duration_s:.3f} s")
    print(f"Average sample rate: {average_hz:.2f} Hz")
    print(f"Playback speed: {args.speed:g}x")
    print("Keys: Space=pause/resume, R=restart, Q=quit")

    frame_period_s = 1.0 / args.viewer_hz
    elapsed_recording_us = 0.0
    displayed_sample_index = -1
    paused = False
    finished = False
    should_quit = False

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
    ) as viewer:
        previous_wall_time = time.monotonic()

        while viewer.is_running() and not should_quit:
            while True:
                try:
                    command = commands.get_nowait()
                except queue.Empty:
                    break

                if command == "quit":
                    should_quit = True
                elif command == "restart":
                    elapsed_recording_us = 0.0
                    displayed_sample_index = -1
                    paused = False
                    finished = False
                    print("Playback restarted")
                elif command == "toggle_pause":
                    if finished:
                        continue
                    paused = not paused
                    print(
                        "Playback paused"
                        if paused
                        else "Playback resumed"
                    )

            now = time.monotonic()
            wall_delta_s = now - previous_wall_time
            previous_wall_time = now

            if not paused and not should_quit:
                elapsed_recording_us += (
                    wall_delta_s
                    * args.speed
                    * 1_000_000
                )

                if elapsed_recording_us >= duration_us:
                    if args.loop and duration_us > 0:
                        elapsed_recording_us %= duration_us
                        displayed_sample_index = -1
                    else:
                        elapsed_recording_us = float(
                            duration_us
                        )
                        paused = True
                        if not finished:
                            print(
                                "Playback finished; press R "
                                "to replay or Q to quit"
                            )
                        finished = True

            sample_index = bisect.bisect_right(
                relative_timestamps_us,
                elapsed_recording_us,
            ) - 1
            sample_index = max(
                0,
                min(sample_index, len(samples) - 1),
            )

            if sample_index != displayed_sample_index:
                with viewer.lock():
                    set_mujoco_joint_angles(
                        model,
                        data,
                        joint_addresses,
                        samples[
                            sample_index
                        ].joint_angles_rad,
                    )
                displayed_sample_index = sample_index

            viewer.sync()

            remaining_frame_time = (
                frame_period_s
                - (time.monotonic() - now)
            )
            if remaining_frame_time > 0.0:
                time.sleep(remaining_frame_time)


if __name__ == "__main__":
    main()
