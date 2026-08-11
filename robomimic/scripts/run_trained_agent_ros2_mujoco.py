#!/usr/bin/env python3

import argparse
import threading
import time
from collections import deque

import numpy as np
import torch
from PIL import Image as PILImage

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float64MultiArray

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils


def ros_image_to_rgb(msg):
    h = int(msg.height)
    w = int(msg.width)
    enc = msg.encoding.lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("rgb8", "bgr8"):
        row = int(msg.step) if msg.step else w * 3
        img = data.reshape(h, row)[:, :w * 3].reshape(h, w, 3)
        if enc == "bgr8":
            img = img[..., ::-1]

    elif enc in ("rgba8", "bgra8"):
        row = int(msg.step) if msg.step else w * 4
        img = data.reshape(h, row)[:, :w * 4].reshape(h, w, 4)[..., :3]
        if enc == "bgra8":
            img = img[..., ::-1]

    elif enc in ("mono8", "8uc1"):
        row = int(msg.step) if msg.step else w
        img = data.reshape(h, row)[:, :w]
        img = np.repeat(img[..., None], 3, axis=-1)

    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    return np.ascontiguousarray(img, dtype=np.uint8)


def resize_image(img, expected_shape):
    height = int(expected_shape[-2])
    width = int(expected_shape[-1])

    if img.shape[:2] == (height, width):
        return img

    img = PILImage.fromarray(img).resize((width, height), PILImage.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


class RobomimicROS2Inference(Node):

    SUPPORTED_OBS = {
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_joint_pos",
        "robot0_joint_vel",
        "robot0_gripper_qpos",
        "robot0_gripper_qvel",
        "agentview_image",
        "robot0_eye_in_hand_image",
        "object",
    }

    def __init__(self, args):
        super().__init__("robomimic_inference")

        self.args = args
        self.data_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.callback_group = ReentrantCallbackGroup()

        device = TorchUtils.get_torch_device(try_to_use_cuda=True)

        self.policy, self.ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=args.agent,
            device=device,
            verbose=True,
        )

        self.policy.start_episode()

        shape_meta = self.ckpt_dict["shape_metadata"]
        self.required_obs = list(shape_meta["all_obs_keys"])
        self.obs_shapes = {key: tuple(value) for key, value in shape_meta["all_shapes"].items()}
        self.action_dim = int(shape_meta["ac_dim"])

        self.frame_stack = 2
        self.obs_history = deque(maxlen=self.frame_stack)

        unsupported = [key for key in self.required_obs if key not in self.SUPPORTED_OBS]
        if unsupported:
            raise RuntimeError(f"Unsupported observation keys: {unsupported}")

        expected_action_dim = 14 if args.arm == "dual" else 7
        if self.action_dim != expected_action_dim:
            raise RuntimeError(
                f"Checkpoint action dimension mismatch: arm={args.arm}, "
                f"checkpoint={self.action_dim}, expected={expected_action_dim}"
            )

        self.left_arm_joints = [f"left_fr3_joint{i}" for i in range(1, 8)]
        self.right_arm_joints = [f"right_fr3_joint{i}" for i in range(1, 8)]

        self.left_gripper_joints = ["left_fr3_finger_joint1", "left_fr3_finger_joint2"]
        self.right_gripper_joints = ["right_fr3_finger_joint1", "right_fr3_finger_joint2"]

        if args.arm == "left":
            self.arm_joints = self.left_arm_joints
            self.gripper_joints = self.left_gripper_joints
        elif args.arm == "right":
            self.arm_joints = self.right_arm_joints
            self.gripper_joints = self.right_gripper_joints
        else:
            self.arm_joints = self.left_arm_joints + self.right_arm_joints
            self.gripper_joints = self.left_gripper_joints + self.right_gripper_joints

        self.left_eef_pos = None
        self.left_eef_quat = None
        self.right_eef_pos = None
        self.right_eef_quat = None

        self.left_eef_time = 0.0
        self.right_eef_time = 0.0

        self.latest_obs = {key: None for key in self.required_obs}
        self.last_update = {key: 0.0 for key in self.required_obs}

        self.inference_active = False
        self.session_id = 0
        self.step = 0
        self.last_warning_time = 0.0

        self.get_logger().info(f"Device: {device}")
        self.get_logger().info(f"Arm: {args.arm}")
        self.get_logger().info(f"Required observations: {self.required_obs}")
        self.get_logger().info(f"Observation shapes: {self.obs_shapes}")
        self.get_logger().info(f"Frame stack: {self.frame_stack}")
        self.get_logger().info(f"Absolute action dimension: {self.action_dim}")

        action_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        active_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.action_pub = self.create_publisher(Float64MultiArray, args.action_topic, action_qos)

        self.create_subscription(
            Bool,
            args.active_topic,
            self.active_callback,
            active_qos,
            callback_group=self.callback_group,
        )

        if "robot0_eef_pos" in self.required_obs or "robot0_eef_quat" in self.required_obs:
            if args.arm in ("left", "dual"):
                self.create_subscription(
                    PoseStamped,
                    args.left_eef_topic,
                    self.left_eef_callback,
                    qos_profile_sensor_data,
                    callback_group=self.callback_group,
                )

            if args.arm in ("right", "dual"):
                self.create_subscription(
                    PoseStamped,
                    args.right_eef_topic,
                    self.right_eef_callback,
                    qos_profile_sensor_data,
                    callback_group=self.callback_group,
                )

        joint_obs = {
            "robot0_joint_pos",
            "robot0_joint_vel",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel",
        }

        if any(key in self.required_obs for key in joint_obs):
            self.create_subscription(
                JointState,
                args.joint_topic,
                self.joint_callback,
                qos_profile_sensor_data,
                callback_group=self.callback_group,
            )

        if "agentview_image" in self.required_obs:
            self.create_subscription(
                Image,
                args.agentview_topic,
                lambda msg: self.image_callback(msg, "agentview_image"),
                qos_profile_sensor_data,
                callback_group=self.callback_group,
            )

        if "robot0_eye_in_hand_image" in self.required_obs:
            self.create_subscription(
                Image,
                args.eye_in_hand_topic,
                lambda msg: self.image_callback(msg, "robot0_eye_in_hand_image"),
                qos_profile_sensor_data,
                callback_group=self.callback_group,
            )

        if "object" in self.required_obs:
            self.create_subscription(
                Float64MultiArray,
                args.object_topic,
                self.object_callback,
                qos_profile_sensor_data,
                callback_group=self.callback_group,
            )

        self.timer = self.create_timer(
            1.0 / args.rate,
            self.inference_callback,
            callback_group=self.callback_group,
        )

        if args.arm in ("left", "dual"):
            self.get_logger().info(f"Left EEF pose: {args.left_eef_topic}")

        if args.arm in ("right", "dual"):
            self.get_logger().info(f"Right EEF pose: {args.right_eef_topic}")

        self.get_logger().info(f"Joint states: {args.joint_topic}")
        self.get_logger().info(f"Agentview: {args.agentview_topic}")
        self.get_logger().info(f"Eye-in-hand: {args.eye_in_hand_topic}")
        self.get_logger().info(f"Absolute action: {args.action_topic}")

    def set_observation(self, key, value):
        if key not in self.latest_obs:
            return

        with self.data_lock:
            self.latest_obs[key] = np.asarray(value).copy()
            self.last_update[key] = time.monotonic()

    def clear_observations(self):
        with self.data_lock:
            for key in self.required_obs:
                self.latest_obs[key] = None
                self.last_update[key] = 0.0

            self.left_eef_pos = None
            self.left_eef_quat = None
            self.right_eef_pos = None
            self.right_eef_quat = None
            self.left_eef_time = 0.0
            self.right_eef_time = 0.0

        self.obs_history.clear()

    def active_callback(self, msg):
        active = bool(msg.data)

        if active == self.inference_active:
            return

        self.session_id += 1
        self.inference_active = active

        if active:
            self.clear_observations()
            self.policy.start_episode()
            self.step = 0
            self.get_logger().info("Inference started")
        else:
            self.obs_history.clear()
            self.get_logger().info("Inference stopped")

    def pose_msg_to_arrays(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation

        pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float32)

        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            return None, None

        quat /= norm
        return pos, quat

    def update_eef_observation(self):
        if self.args.arm == "left":
            if self.left_eef_pos is None or self.left_eef_quat is None:
                return

            self.latest_obs["robot0_eef_pos"] = self.left_eef_pos.copy()
            self.latest_obs["robot0_eef_quat"] = self.left_eef_quat.copy()

            if "robot0_eef_pos" in self.last_update:
                self.last_update["robot0_eef_pos"] = self.left_eef_time
            if "robot0_eef_quat" in self.last_update:
                self.last_update["robot0_eef_quat"] = self.left_eef_time

        elif self.args.arm == "right":
            if self.right_eef_pos is None or self.right_eef_quat is None:
                return

            self.latest_obs["robot0_eef_pos"] = self.right_eef_pos.copy()
            self.latest_obs["robot0_eef_quat"] = self.right_eef_quat.copy()

            if "robot0_eef_pos" in self.last_update:
                self.last_update["robot0_eef_pos"] = self.right_eef_time
            if "robot0_eef_quat" in self.last_update:
                self.last_update["robot0_eef_quat"] = self.right_eef_time

        else:
            if (
                self.left_eef_pos is None
                or self.left_eef_quat is None
                or self.right_eef_pos is None
                or self.right_eef_quat is None
            ):
                return

            self.latest_obs["robot0_eef_pos"] = np.concatenate(
                [self.left_eef_pos, self.right_eef_pos], axis=0
            )

            self.latest_obs["robot0_eef_quat"] = np.concatenate(
                [self.left_eef_quat, self.right_eef_quat], axis=0
            )

            update_time = min(self.left_eef_time, self.right_eef_time)

            if "robot0_eef_pos" in self.last_update:
                self.last_update["robot0_eef_pos"] = update_time
            if "robot0_eef_quat" in self.last_update:
                self.last_update["robot0_eef_quat"] = update_time

    def left_eef_callback(self, msg):
        pos, quat = self.pose_msg_to_arrays(msg)
        if pos is None:
            return

        with self.data_lock:
            self.left_eef_pos = pos
            self.left_eef_quat = quat
            self.left_eef_time = time.monotonic()
            self.update_eef_observation()

    def right_eef_callback(self, msg):
        pos, quat = self.pose_msg_to_arrays(msg)
        if pos is None:
            return

        with self.data_lock:
            self.right_eef_pos = pos
            self.right_eef_quat = quat
            self.right_eef_time = time.monotonic()
            self.update_eef_observation()

    def joint_callback(self, msg):
        position = dict(zip(msg.name, msg.position))
        velocity = dict(zip(msg.name, msg.velocity)) if len(msg.velocity) == len(msg.name) else {}

        if "robot0_joint_pos" in self.required_obs and all(name in position for name in self.arm_joints):
            value = np.array([position[name] for name in self.arm_joints], dtype=np.float32)
            self.set_observation("robot0_joint_pos", value)

        if "robot0_joint_vel" in self.required_obs and all(name in velocity for name in self.arm_joints):
            value = np.array([velocity[name] for name in self.arm_joints], dtype=np.float32)
            self.set_observation("robot0_joint_vel", value)

        if "robot0_gripper_qpos" in self.required_obs and all(name in position for name in self.gripper_joints):
            value = np.array([position[name] for name in self.gripper_joints], dtype=np.float32)
            self.set_observation("robot0_gripper_qpos", value)

        if "robot0_gripper_qvel" in self.required_obs and all(name in velocity for name in self.gripper_joints):
            value = np.array([velocity[name] for name in self.gripper_joints], dtype=np.float32)
            self.set_observation("robot0_gripper_qvel", value)

    def image_callback(self, msg, key):
        try:
            image = ros_image_to_rgb(msg)
            image = resize_image(image, self.obs_shapes[key])
            self.set_observation(key, image)

        except Exception as error:
            self.get_logger().error(f"{key}: {error}")

    def object_callback(self, msg):
        value = np.asarray(msg.data, dtype=np.float32)
        expected_shape = self.obs_shapes["object"]

        if value.shape != expected_shape:
            self.get_logger().error(
                f"object shape mismatch: received {value.shape}, expected {expected_shape}"
            )
            return

        self.set_observation("object", value)

    def get_observation(self):
        now = time.monotonic()

        with self.data_lock:
            missing = [key for key in self.required_obs if self.latest_obs[key] is None]

            stale = [
                key
                for key in self.required_obs
                if self.latest_obs[key] is not None
                and now - self.last_update[key] > self.args.max_obs_age
            ]

            if missing or stale:
                if now - self.last_warning_time > 1.0:
                    if missing:
                        self.get_logger().warning(f"Missing observations: {missing}")
                    if stale:
                        self.get_logger().warning(f"Stale observations: {stale}")

                    self.last_warning_time = now

                return None

            return {key: self.latest_obs[key].copy() for key in self.required_obs}

    def update_observation_history(self, obs):
        obs_copy = {key: value.copy() for key, value in obs.items()}

        if len(self.obs_history) == 0:
            for _ in range(self.frame_stack):
                self.obs_history.append({key: value.copy() for key, value in obs_copy.items()})
        else:
            self.obs_history.append(obs_copy)

        return {
            key: np.stack([history[key] for history in self.obs_history], axis=0)
            for key in self.required_obs
        }

    def inference_callback(self):
        if not self.inference_active:
            return

        if not self.inference_lock.acquire(blocking=False):
            return

        session_id = self.session_id

        try:
            obs = self.get_observation()
            if obs is None:
                return

            stacked_obs = self.update_observation_history(obs)

            start = time.perf_counter()

            with torch.inference_mode():
                output = self.policy(ob=stacked_obs)

            action = output[0] if isinstance(output, tuple) else output
            action = np.asarray(action, dtype=np.float64).reshape(-1)

            if action.size != self.action_dim:
                raise RuntimeError(f"Action dimension mismatch: {action.size} != {self.action_dim}")

            if not np.all(np.isfinite(action)):
                raise RuntimeError(f"Invalid action: {action}")

            if not self.inference_active or session_id != self.session_id:
                return

            msg = Float64MultiArray()
            msg.data = action.tolist()
            self.action_pub.publish(msg)

            if self.step % max(1, int(self.args.rate)) == 0:
                inference_ms = (time.perf_counter() - start) * 1000.0
                self.get_logger().info(
                    f"step={self.step}, inference={inference_ms:.1f} ms, "
                    f"absolute_action={np.array2string(action, precision=4)}"
                )

                shapes = {key: value.shape for key, value in stacked_obs.items()}
                self.get_logger().info(f"Stacked observation shapes: {shapes}")

            self.step += 1

        except Exception as error:
            self.get_logger().error(f"Inference failed: {error}")

        finally:
            self.inference_lock.release()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--agent", type=str, required=True)
    parser.add_argument("--arm", choices=["left", "right", "dual"], required=True)

    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--max_obs_age", type=float, default=0.5)

    parser.add_argument("--active_topic", type=str, default="/robomimic/inference_active")
    parser.add_argument("--action_topic", type=str, default="/robomimic/absolute_action")

    parser.add_argument("--left_eef_topic", type=str, default="/robomimic/obs/left_eef_pose")
    parser.add_argument("--right_eef_topic", type=str, default="/robomimic/obs/right_eef_pose")
    parser.add_argument("--joint_topic", type=str, default="/joint_states")

    parser.add_argument(
        "--agentview_topic",
        type=str,
        default="/mujoco_ros_hardware/top_azure/color/image_raw",
    )

    parser.add_argument(
        "--eye_in_hand_topic",
        type=str,
        default="/mujoco_ros_hardware/right_d435i/color/image_raw",
    )

    parser.add_argument("--object_topic", type=str, default="/robomimic/obs/object")

    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()

    node = RobomimicROS2Inference(args)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()

    except KeyboardInterrupt:
        pass

    finally:
        executor.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()