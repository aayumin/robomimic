#!/usr/bin/env python3

import argparse
import threading
import time
import traceback
from collections import deque

import cv2
import numpy as np
import torch

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
    qos_profile_sensor_data,
)

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, Float64MultiArray

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils


# ============================================================
# Utility functions
# ============================================================

def compressed_image_to_rgb(msg: CompressedImage):
    """
    Convert sensor_msgs/CompressedImage to uint8 RGB image.

    Output:
        np.ndarray with shape (H, W, 3)
        dtype uint8
    """

    if msg.data is None or len(msg.data) == 0:
        raise ValueError("CompressedImage contains no data")

    data = np.frombuffer(msg.data, dtype=np.uint8)

    image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError("cv2.imdecode() failed")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)

    if image_rgb.ndim != 3:
        raise ValueError(
            f"Decoded image must be 3D, got shape={image_rgb.shape}"
        )

    if image_rgb.shape[-1] != 3:
        raise ValueError(
            f"Decoded image must be HWC with 3 channels, "
            f"got shape={image_rgb.shape}"
        )

    return image_rgb


def resize_rgb_image(image, expected_shape):
    """
    Resize RGB image to the spatial resolution expected by checkpoint.

    IMPORTANT:
    robomimic raw RGB observations are expected as:

        (H, W, 3)

    while checkpoint metadata may store image shape as:

        (3, H, W)

    or:

        (H, W, 3)

    """

    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(
            f"RGB image must have 3 dimensions, got {image.shape}"
        )

    if image.shape[-1] != 3:
        raise ValueError(
            f"RGB image must be HWC with 3 channels, got {image.shape}"
        )

    expected_shape = tuple(int(x) for x in expected_shape)

    # --------------------------------------------------------
    # Determine expected H/W
    # --------------------------------------------------------

    if len(expected_shape) != 3:
        raise ValueError(
            f"Expected RGB shape must have 3 dimensions, "
            f"got {expected_shape}"
        )

    # Common robomimic checkpoint format:
    # (C, H, W)
    if expected_shape[0] == 3:
        expected_h = expected_shape[1]
        expected_w = expected_shape[2]

    # Alternative:
    # (H, W, C)
    elif expected_shape[2] == 3:
        expected_h = expected_shape[0]
        expected_w = expected_shape[1]

    else:
        raise ValueError(
            f"Cannot determine RGB channel dimension from "
            f"checkpoint shape {expected_shape}"
        )

    if image.shape[:2] != (expected_h, expected_w):
        image = cv2.resize(
            image,
            (expected_w, expected_h),
            interpolation=cv2.INTER_AREA,
        )

    image = np.ascontiguousarray(image, dtype=np.uint8)

    # Final sanity check
    if image.shape != (expected_h, expected_w, 3):
        raise RuntimeError(
            f"Final RGB image shape mismatch: "
            f"{image.shape} != {(expected_h, expected_w, 3)}"
        )

    return image


# ============================================================
# Main ROS2 node
# ============================================================

class RobomimicROS2Inference(Node):

    FRAME_STACK = 2

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

    IMAGE_OBS = {
        "agentview_image",
        "robot0_eye_in_hand_image",
    }

    def __init__(self, args):

        super().__init__("robomimic_inference")

        self.args = args

        self.data_lock = threading.Lock()
        self.inference_lock = threading.Lock()

        self.callback_group = ReentrantCallbackGroup()

        # ====================================================
        # Device
        # ====================================================

        self.device = TorchUtils.get_torch_device(
            try_to_use_cuda=True
        )

        self.get_logger().info(
            f"Device: {self.device}"
        )

        if torch.cuda.is_available():

            self.get_logger().info(
                f"CUDA device: {torch.cuda.get_device_name(0)}"
            )

            self.get_logger().info(
                f"CUDA version: {torch.version.cuda}"
            )

        else:
            self.get_logger().warning(
                "CUDA is NOT available. Inference will run on CPU."
            )

        # ====================================================
        # Load policy
        # ====================================================

        self.get_logger().info(
            f"Loading checkpoint: {args.agent}"
        )

        self.policy, self.ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=args.agent,
            device=self.device,
            verbose=True,
        )

        self.policy.start_episode()

        # ====================================================
        # Checkpoint metadata
        # ====================================================

        shape_meta = self.ckpt_dict["shape_metadata"]

        self.required_obs = list(
            shape_meta["all_obs_keys"]
        )

        self.obs_shapes = {
            key: tuple(value)
            for key, value in shape_meta["all_shapes"].items()
        }

        self.action_dim = int(
            shape_meta["ac_dim"]
        )

        self.get_logger().info(
            f"Policy type: {type(self.policy)}"
        )

        self.get_logger().info(
            f"Required observations: {self.required_obs}"
        )

        self.get_logger().info(
            f"Observation shapes: {self.obs_shapes}"
        )

        self.get_logger().info(
            f"Action dimension: {self.action_dim}"
        )

        # ====================================================
        # Frame stack
        # ====================================================

        self.frame_stack = self.FRAME_STACK

        self.obs_history = deque(
            maxlen=self.frame_stack
        )

        self.get_logger().info(
            f"Frame stack: {self.frame_stack}"
        )

        # ====================================================
        # Validate observation keys
        # ====================================================

        unsupported = [
            key
            for key in self.required_obs
            if key not in self.SUPPORTED_OBS
        ]

        if unsupported:
            raise RuntimeError(
                f"Unsupported observation keys: {unsupported}"
            )

        # ====================================================
        # Validate image observations
        # ====================================================

        for key in self.IMAGE_OBS:

            if key not in self.required_obs:
                continue

            shape = self.obs_shapes[key]

            self.get_logger().info(
                f"Image observation: {key}, "
                f"checkpoint shape={shape}"
            )

            if len(shape) != 3:
                raise RuntimeError(
                    f"{key} must be a 3D RGB image shape, "
                    f"got {shape}"
                )

            if not (
                shape[0] == 3
                or shape[2] == 3
            ):
                raise RuntimeError(
                    f"{key}: cannot identify RGB channel "
                    f"in checkpoint shape {shape}"
                )

        # ====================================================
        # Action dimension validation
        # ====================================================

        expected_action_dim = (
            16 if args.arm == "dual" else 8
        )

        if self.action_dim != expected_action_dim:

            raise RuntimeError(
                f"Checkpoint action dimension mismatch: "
                f"arm={args.arm}, "
                f"checkpoint={self.action_dim}, "
                f"expected={expected_action_dim}"
            )

        # ====================================================
        # Timing
        # ====================================================

        self.action_period = 1.0 / args.rate

        self.next_action_time = 0.0

        self.get_logger().info(
            f"Inference rate: {args.rate:.1f} Hz"
        )

        # ====================================================
        # Joint names
        # ====================================================

        self.left_arm_joints = [
            f"left_fr3_joint{i}"
            for i in range(1, 8)
        ]

        self.right_arm_joints = [
            f"right_fr3_joint{i}"
            for i in range(1, 8)
        ]

        self.left_gripper_joints = [
            "left_fr3_finger_joint1",
            "left_fr3_finger_joint2",
        ]

        self.right_gripper_joints = [
            "right_fr3_finger_joint1",
            "right_fr3_finger_joint2",
        ]

        if args.arm == "left":

            self.arm_joints = (
                self.left_arm_joints
            )

            self.gripper_joints = (
                self.left_gripper_joints
            )

        elif args.arm == "right":

            self.arm_joints = (
                self.right_arm_joints
            )

            self.gripper_joints = (
                self.right_gripper_joints
            )

        else:

            self.arm_joints = (
                self.left_arm_joints
                + self.right_arm_joints
            )

            self.gripper_joints = (
                self.left_gripper_joints
                + self.right_gripper_joints
            )

        # ====================================================
        # EEF states
        # ====================================================

        self.left_eef_pos = None
        self.left_eef_quat = None

        self.right_eef_pos = None
        self.right_eef_quat = None

        self.left_eef_time = 0.0
        self.right_eef_time = 0.0

        # ====================================================
        # Observation buffers
        # ====================================================

        self.latest_obs = {
            key: None
            for key in self.required_obs
        }

        self.last_update = {
            key: 0.0
            for key in self.required_obs
        }

        # ====================================================
        # Runtime state
        # ====================================================

        self.inference_active = False

        self.session_id = 0

        self.step = 0

        self.last_warning_time = 0.0

        self.last_debug_time = 0.0

        # ====================================================
        # QoS
        # ====================================================

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

        # ====================================================
        # Publishers
        # ====================================================

        self.action_pub = self.create_publisher(
            Float64MultiArray,
            args.action_topic,
            action_qos,
        )

        # ====================================================
        # Active topic
        # ====================================================

        self.create_subscription(
            Bool,
            args.active_topic,
            self.active_callback,
            active_qos,
            callback_group=self.callback_group,
        )

        # ====================================================
        # EEF subscriptions
        # ====================================================

        if (
            "robot0_eef_pos" in self.required_obs
            or
            "robot0_eef_quat" in self.required_obs
        ):

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

        # ====================================================
        # Joint subscription
        # ====================================================

        joint_obs = {
            "robot0_joint_pos",
            "robot0_joint_vel",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel",
        }

        if any(
            key in self.required_obs
            for key in joint_obs
        ):

            self.create_subscription(
                JointState,
                args.joint_topic,
                self.joint_callback,
                qos_profile_sensor_data,
                callback_group=self.callback_group,
            )

        # ====================================================
        # Camera
        #
        # IMPORTANT:
        #
        # agentview_image and robot0_eye_in_hand_image
        # intentionally use the SAME ROS topic.
        #
        # The same decoded image is assigned to both
        # observation keys.
        # ====================================================

        if (
            "agentview_image" in self.required_obs
            or
            "robot0_eye_in_hand_image" in self.required_obs
        ):

            self.create_subscription(
                CompressedImage,
                args.image_topic,
                self.camera_callback,
                qos_profile_sensor_data,
                callback_group=self.callback_group,
            )

        # ====================================================
        # Object observation
        # ====================================================

        if "object" in self.required_obs:

            self.create_subscription(
                Float64MultiArray,
                args.object_topic,
                self.object_callback,
                qos_profile_sensor_data,
                callback_group=self.callback_group,
            )

        # ====================================================
        # Inference timer
        # ====================================================

        self.timer = self.create_timer(
            self.action_period,
            self.inference_callback,
            callback_group=self.callback_group,
        )

        # ====================================================
        # Print configuration
        # ====================================================

        self.get_logger().info(
            f"Left EEF pose: {args.left_eef_topic}"
        )

        self.get_logger().info(
            f"Right EEF pose: {args.right_eef_topic}"
        )

        self.get_logger().info(
            f"Joint states: {args.joint_topic}"
        )

        self.get_logger().info(
            f"Camera: {args.image_topic}"
        )

        self.get_logger().info(
            "Agentview and Eye-in-hand use the SAME camera topic"
        )

        self.get_logger().info(
            f"Absolute action: {args.action_topic}"
        )

        self.get_logger().info(
            "====================================="
        )

    # ========================================================
    # Observation handling
    # ========================================================

    def set_observation(self, key, value):

        if key not in self.latest_obs:
            return

        with self.data_lock:

            self.latest_obs[key] = (
                np.asarray(value).copy()
            )

            self.last_update[key] = (
                time.monotonic()
            )

    # ========================================================

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

    # ========================================================
    # Active callback
    # ========================================================

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

            self.next_action_time = (
                time.monotonic()
            )

            self.get_logger().info(
                "Inference started"
            )

        else:

            self.obs_history.clear()

            self.get_logger().info(
                "Inference stopped"
            )

    # ========================================================
    # EEF
    # ========================================================

    def pose_msg_to_arrays(self, msg):

        p = msg.pose.position

        q = msg.pose.orientation

        pos = np.array(
            [
                p.x,
                p.y,
                p.z,
            ],
            dtype=np.float32,
        )

        quat = np.array(
            [
                q.x,
                q.y,
                q.z,
                q.w,
            ],
            dtype=np.float32,
        )

        norm = np.linalg.norm(quat)

        if norm < 1e-8:
            return None, None

        quat /= norm

        return pos, quat

    # ========================================================

    def update_eef_observation(self):

        if self.args.arm == "left":

            if (
                self.left_eef_pos is None
                or
                self.left_eef_quat is None
            ):
                return

            self.latest_obs[
                "robot0_eef_pos"
            ] = self.left_eef_pos.copy()

            self.latest_obs[
                "robot0_eef_quat"
            ] = self.left_eef_quat.copy()

            if (
                "robot0_eef_pos"
                in self.last_update
            ):

                self.last_update[
                    "robot0_eef_pos"
                ] = self.left_eef_time

            if (
                "robot0_eef_quat"
                in self.last_update
            ):

                self.last_update[
                    "robot0_eef_quat"
                ] = self.left_eef_time

        elif self.args.arm == "right":

            if (
                self.right_eef_pos is None
                or
                self.right_eef_quat is None
            ):
                return

            self.latest_obs[
                "robot0_eef_pos"
            ] = self.right_eef_pos.copy()

            self.latest_obs[
                "robot0_eef_quat"
            ] = self.right_eef_quat.copy()

            if (
                "robot0_eef_pos"
                in self.last_update
            ):

                self.last_update[
                    "robot0_eef_pos"
                ] = self.right_eef_time

            if (
                "robot0_eef_quat"
                in self.last_update
            ):

                self.last_update[
                    "robot0_eef_quat"
                ] = self.right_eef_time

        else:

            if (
                self.left_eef_pos is None
                or
                self.left_eef_quat is None
                or
                self.right_eef_pos is None
                or
                self.right_eef_quat is None
            ):
                return

            self.latest_obs[
                "robot0_eef_pos"
            ] = np.concatenate(
                [
                    self.left_eef_pos,
                    self.right_eef_pos,
                ],
                axis=0,
            )

            self.latest_obs[
                "robot0_eef_quat"
            ] = np.concatenate(
                [
                    self.left_eef_quat,
                    self.right_eef_quat,
                ],
                axis=0,
            )

            update_time = min(
                self.left_eef_time,
                self.right_eef_time,
            )

            if (
                "robot0_eef_pos"
                in self.last_update
            ):

                self.last_update[
                    "robot0_eef_pos"
                ] = update_time

            if (
                "robot0_eef_quat"
                in self.last_update
            ):

                self.last_update[
                    "robot0_eef_quat"
                ] = update_time

    # ========================================================

    def left_eef_callback(self, msg):

        pos, quat = (
            self.pose_msg_to_arrays(msg)
        )

        if pos is None:
            return

        with self.data_lock:

            self.left_eef_pos = pos

            self.left_eef_quat = quat

            self.left_eef_time = (
                time.monotonic()
            )

            self.update_eef_observation()

    # ========================================================

    def right_eef_callback(self, msg):

        pos, quat = (
            self.pose_msg_to_arrays(msg)
        )

        if pos is None:
            return

        with self.data_lock:

            self.right_eef_pos = pos

            self.right_eef_quat = quat

            self.right_eef_time = (
                time.monotonic()
            )

            self.update_eef_observation()

    # ========================================================
    # Joint states
    # ========================================================

    def joint_callback(self, msg):

        position = dict(
            zip(
                msg.name,
                msg.position,
            )
        )

        velocity = {}

        if (
            len(msg.velocity)
            == len(msg.name)
        ):

            velocity = dict(
                zip(
                    msg.name,
                    msg.velocity,
                )
            )

        if (
            "robot0_joint_pos"
            in self.required_obs
            and
            all(
                name in position
                for name in self.arm_joints
            )
        ):

            value = np.array(
                [
                    position[name]
                    for name in self.arm_joints
                ],
                dtype=np.float32,
            )

            self.set_observation(
                "robot0_joint_pos",
                value,
            )

        if (
            "robot0_joint_vel"
            in self.required_obs
            and
            all(
                name in velocity
                for name in self.arm_joints
            )
        ):

            value = np.array(
                [
                    velocity[name]
                    for name in self.arm_joints
                ],
                dtype=np.float32,
            )

            self.set_observation(
                "robot0_joint_vel",
                value,
            )

        if (
            "robot0_gripper_qpos"
            in self.required_obs
            and
            all(
                name in position
                for name in self.gripper_joints
            )
        ):

            value = np.array(
                [
                    position[name]
                    for name in self.gripper_joints
                ],
                dtype=np.float32,
            )

            self.set_observation(
                "robot0_gripper_qpos",
                value,
            )

        if (
            "robot0_gripper_qvel"
            in self.required_obs
            and
            all(
                name in velocity
                for name in self.gripper_joints
            )
        ):

            value = np.array(
                [
                    velocity[name]
                    for name in self.gripper_joints
                ],
                dtype=np.float32,
            )

            self.set_observation(
                "robot0_gripper_qvel",
                value,
            )

    # ========================================================
    # Camera callback
    # ========================================================

    def camera_callback(self, msg):

        try:

            image = compressed_image_to_rgb(msg)

            # ------------------------------------------------
            # agentview
            # ------------------------------------------------

            if (
                "agentview_image"
                in self.required_obs
            ):

                agentview = resize_rgb_image(
                    image,
                    self.obs_shapes[
                        "agentview_image"
                    ],
                )

                self.set_observation(
                    "agentview_image",
                    agentview,
                )

            # ------------------------------------------------
            # eye-in-hand
            #
            # Intentionally SAME image.
            # ------------------------------------------------

            if (
                "robot0_eye_in_hand_image"
                in self.required_obs
            ):

                eye_in_hand = resize_rgb_image(
                    image,
                    self.obs_shapes[
                        "robot0_eye_in_hand_image"
                    ],
                )

                self.set_observation(
                    "robot0_eye_in_hand_image",
                    eye_in_hand,
                )

        except Exception as error:

            self.get_logger().error(
                "Camera callback failed: "
                f"type={type(error).__name__}, "
                f"repr={repr(error)}"
            )

    # ========================================================
    # Object
    # ========================================================

    def object_callback(self, msg):

        value = np.asarray(
            msg.data,
            dtype=np.float32,
        )

        expected_shape = self.obs_shapes[
            "object"
        ]

        if value.shape != expected_shape:

            self.get_logger().error(
                f"object shape mismatch: "
                f"received {value.shape}, "
                f"expected {expected_shape}"
            )

            return

        self.set_observation(
            "object",
            value,
        )

    # ========================================================
    # Get observation
    # ========================================================

    def get_observation(self):

        now = time.monotonic()

        with self.data_lock:

            missing = [
                key
                for key in self.required_obs
                if self.latest_obs[key] is None
            ]

            stale = [
                key
                for key in self.required_obs
                if (
                    self.latest_obs[key] is not None
                    and
                    (
                        now
                        - self.last_update[key]
                    )
                    > self.args.max_obs_age
                )
            ]

            if missing or stale:

                if (
                    now
                    - self.last_warning_time
                    > 1.0
                ):

                    if missing:

                        self.get_logger().warning(
                            f"Missing observations: "
                            f"{missing}"
                        )

                    if stale:

                        self.get_logger().warning(
                            f"Stale observations: "
                            f"{stale}"
                        )

                    self.last_warning_time = now

                return None

            return {
                key: self.latest_obs[key].copy()
                for key in self.required_obs
            }

    # ========================================================
    # Frame stack
    # ========================================================

    def update_observation_history(self, obs):

        obs_copy = {
            key: value.copy()
            for key, value in obs.items()
        }

        # ----------------------------------------------------
        # First observation:
        # duplicate it FRAME_STACK times
        # ----------------------------------------------------

        if len(self.obs_history) == 0:

            for _ in range(
                self.frame_stack
            ):

                self.obs_history.append(
                    {
                        key: value.copy()
                        for key, value
                        in obs_copy.items()
                    }
                )

        else:

            self.obs_history.append(
                obs_copy
            )

        # ----------------------------------------------------
        # Stack along FIRST dimension
        #
        # low dim:
        #   (N,) -> (2,N)
        #
        # image:
        #   (H,W,3) -> (2,H,W,3)
        # ----------------------------------------------------

        stacked = {
            key: np.stack(
                [
                    history[key]
                    for history
                    in self.obs_history
                ],
                axis=0,
            )
            for key in self.required_obs
        }

        return stacked

    # ========================================================
    # Debug observation shapes
    # ========================================================

    def debug_print_stacked_shapes(
        self,
        stacked_obs,
    ):

        now = time.monotonic()

        if (
            now
            - self.last_debug_time
            < 2.0
        ):
            return

        self.last_debug_time = now

        self.get_logger().info(
            "========== OBSERVATION SHAPES =========="
        )

        for key in self.required_obs:

            value = stacked_obs[key]

            self.get_logger().info(
                f"STACKED[{key}]: "
                f"shape={value.shape}, "
                f"dtype={value.dtype}"
            )

        self.get_logger().info(
            "========================================="
        )

    # ========================================================
    # Validate before robomimic policy
    # ========================================================

    def validate_stacked_observation(
        self,
        stacked_obs,
    ):

        for key in self.required_obs:

            value = stacked_obs[key]

            if value is None:

                raise RuntimeError(
                    f"Observation {key} is None"
                )

            # ------------------------------------------------
            # Frame stack dimension
            # ------------------------------------------------

            if value.shape[0] != self.frame_stack:

                raise RuntimeError(
                    f"{key}: frame stack mismatch: "
                    f"{value.shape}, "
                    f"expected first dim "
                    f"{self.frame_stack}"
                )

            # ------------------------------------------------
            # Image validation
            # ------------------------------------------------

            if key in self.IMAGE_OBS:

                if value.ndim != 4:

                    raise RuntimeError(
                        f"{key}: stacked RGB image "
                        f"must be 4D "
                        f"(T,H,W,C), "
                        f"got {value.shape}"
                    )

                if value.shape[-1] != 3:

                    raise RuntimeError(
                        f"{key}: RGB channel must "
                        f"be LAST dimension, "
                        f"got {value.shape}"
                    )

                # Check against checkpoint spatial size
                expected = self.obs_shapes[key]

                if expected[0] == 3:

                    expected_h = expected[1]
                    expected_w = expected[2]

                else:

                    expected_h = expected[0]
                    expected_w = expected[1]

                actual_h = value.shape[1]
                actual_w = value.shape[2]

                if (
                    actual_h != expected_h
                    or
                    actual_w != expected_w
                ):

                    raise RuntimeError(
                        f"{key}: image spatial "
                        f"shape mismatch: "
                        f"received "
                        f"{value.shape}, "
                        f"checkpoint="
                        f"{expected}"
                    )


    # ========================================================
    # Action post-processing
    # ========================================================

    def normalize_quat_xyzw(self, quat):
        """
        Normalize quaternion [x, y, z, w].

        Policy output may not be exactly unit-norm, but downstream
        absolute pose control should receive a valid quaternion.
        """
        quat = np.asarray(quat, dtype=np.float64)
        norm = np.linalg.norm(quat)

        if norm < 1e-8 or not np.isfinite(norm):
            # Safe fallback: identity quaternion.
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        quat = quat / norm

        # Optional canonical hemisphere for consistency.
        if quat[3] < 0.0:
            quat = -quat

        return quat.astype(np.float64)

    def postprocess_absolute_action(self, action):
        """
        Postprocess real-world absolute action.

        Action format:
            single arm:
                [x, y, z, qx, qy, qz, qw, gripper]

            dual arm:
                left  [x, y, z, qx, qy, qz, qw, gripper]
                right [x, y, z, qx, qy, qz, qw, gripper]
        """
        action = np.asarray(action, dtype=np.float64).copy()

        single_arm_dim = 8
        num_arms = 2 if self.args.arm == "dual" else 1

        expected_dim = single_arm_dim * num_arms

        if action.size != expected_dim:
            raise RuntimeError(
                f"Absolute action dimension mismatch: "
                f"received={action.size}, expected={expected_dim}"
            )

        for arm_idx in range(num_arms):
            start = arm_idx * single_arm_dim
            quat_start = start + 3
            quat_end = start + 7

            action[quat_start:quat_end] = self.normalize_quat_xyzw(
                action[quat_start:quat_end]
            )

        return action


    
    # ========================================================
    # Inference
    # ========================================================

    def inference_callback(self):

        if not self.inference_active:
            return

        now = time.monotonic()

        if now < self.next_action_time:
            return

        # ----------------------------------------------------
        # Prevent concurrent inference
        # ----------------------------------------------------

        if not self.inference_lock.acquire(
            blocking=False
        ):
            return

        session_id = self.session_id

        try:

            # =================================================
            # Get observation
            # =================================================

            obs = self.get_observation()

            if obs is None:
                return

            # =================================================
            # Frame stack
            # =================================================

            stacked_obs = (
                self.update_observation_history(
                    obs
                )
            )

            # =================================================
            # Validate
            # =================================================

            self.validate_stacked_observation(
                stacked_obs
            )

            self.debug_print_stacked_shapes(
                stacked_obs
            )

            # =================================================
            # Inference timing
            # =================================================

            start_time = (
                time.perf_counter()
            )

            # =================================================
            # GPU inference
            # =================================================

            if (
                self.device.type
                == "cuda"
            ):

                with (
                    torch.inference_mode(),
                    torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                    ),
                ):

                    output = self.policy(
                        ob=stacked_obs
                    )

            else:

                with torch.inference_mode():

                    output = self.policy(
                        ob=stacked_obs
                    )

            # =================================================
            # Extract action
            # =================================================

            action = (
                output[0]
                if isinstance(
                    output,
                    tuple
                )
                else output
            )

            action = np.asarray(
                action,
                dtype=np.float64,
            ).reshape(-1)

            # =================================================
            # Validate action
            # =================================================

            if (
                action.size
                != self.action_dim
            ):

                raise RuntimeError(
                    f"Action dimension mismatch: "
                    f"{action.size} != "
                    f"{self.action_dim}"
                )

            if not np.all(
                np.isfinite(action)
            ):

                raise RuntimeError(
                    f"Invalid action: "
                    f"{action}"
                )
            
            action = self.postprocess_absolute_action(action)

            if not np.all(np.isfinite(action)):
                raise RuntimeError(
                    f"Invalid postprocessed action: "
                    f"{action}"
                )
            # =================================================
            # Session check
            # =================================================

            if (
                not self.inference_active
                or
                session_id
                != self.session_id
            ):

                return

            # =================================================
            # Publish action
            # =================================================

            msg = Float64MultiArray()

            msg.data = action.tolist()

            self.action_pub.publish(
                msg
            )

            # =================================================
            # Timing
            # =================================================

            self.next_action_time = (
                time.monotonic()
                + self.action_period
            )

            inference_ms = (
                time.perf_counter()
                - start_time
            ) * 1000.0

            self.get_logger().info(
                f"step={self.step}, "
                f"inference={inference_ms:.1f} ms, "
                f"action_dim={action.size}"
            )

            self.step += 1

        except Exception as error:

            # =================================================
            # VERY IMPORTANT:
            # Print full exception and traceback.
            # =================================================

            self.get_logger().error(
                "Inference failed: "
                f"type={type(error).__name__}, "
                f"repr={repr(error)}"
            )

            self.get_logger().error(
                traceback.format_exc()
            )

            # ------------------------------------------------
            # Prevent a failed inference from immediately
            # hammering the GPU at maximum speed.
            # ------------------------------------------------

            self.next_action_time = (
                time.monotonic()
                + 0.1
            )

        finally:

            self.inference_lock.release()


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agent",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--arm",
        choices=[
            "left",
            "right",
            "dual",
        ],
        required=True,
    )

    # --------------------------------------------------------
    # 30 Hz is a reasonable starting point for real robot
    # visual inference.
    # --------------------------------------------------------

    parser.add_argument(
        "--rate",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--max_obs_age",
        type=float,
        default=0.5,
    )

    # --------------------------------------------------------
    # ROS topics
    # --------------------------------------------------------

    parser.add_argument(
        "--active_topic",
        type=str,
        default="/robomimic/inference_active",
    )

    parser.add_argument(
        "--action_topic",
        type=str,
        default="/robomimic/absolute_action",
    )

    parser.add_argument(
        "--left_eef_topic",
        type=str,
        default="/robomimic/obs/left_eef_pose",
    )

    parser.add_argument(
        "--right_eef_topic",
        type=str,
        default="/robomimic/obs/right_eef_pose",
    )

    parser.add_argument(
        "--joint_topic",
        type=str,
        default="/joint_states",
    )

    # --------------------------------------------------------
    # SAME compressed camera topic is intentionally used for:
    #
    #   agentview_image
    #   robot0_eye_in_hand_image
    #
    # because the training dataset used the same image for
    # both modalities.
    # --------------------------------------------------------

    parser.add_argument(
        "--image_topic",
        type=str,
        default="/camera/camera/color/image_raw/compressed",
    )

    parser.add_argument(
        "--object_topic",
        type=str,
        default="/robomimic/obs/object",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    rclpy.init()

    node = None

    try:

        node = RobomimicROS2Inference(
            args
        )

        executor = MultiThreadedExecutor(
            num_threads=4
        )

        executor.add_node(node)

        node.get_logger().info(
            "ROS2 inference executor started"
        )

        executor.spin()

    except KeyboardInterrupt:

        pass

    except Exception as error:

        print(
            "Fatal error:",
            type(error).__name__,
            repr(error),
        )

        print(
            traceback.format_exc()
        )

    finally:

        if node is not None:

            try:
                node.destroy_node()
            except Exception:
                pass

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == "__main__":
    main()