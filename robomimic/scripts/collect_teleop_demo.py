import h5py
import numpy as np
import math

import os
import json
import argparse
from copy import deepcopy
from tqdm import tqdm


import socket
import time
from dataclasses import dataclass, field
import threading
from scipy.spatial.transform import Rotation as R

from pynput import keyboard

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase




def get_camera_info(
    env,
    camera_names=None, 
    camera_height=84, 
    camera_width=84,
):
    """
    Helper function to get camera intrinsics and extrinsics for cameras being used for observations.
    """

    # TODO: make this function more general than just robosuite environments
    assert EnvUtils.is_robosuite_env(env=env)

    # check for v1.5+ robosuite
    import robosuite
    is_v15 = (robosuite.__version__.split(".")[0] == "1") and (robosuite.__version__.split(".")[1] >= "5")

    if camera_names is None:
        return None

    camera_info = dict()
    for cam_name in camera_names:
        K = env.get_camera_intrinsic_matrix(camera_name=cam_name, camera_height=camera_height, camera_width=camera_width)
        R = env.get_camera_extrinsic_matrix(camera_name=cam_name) # camera pose in world frame
        if "eye_in_hand" in cam_name:
            # convert extrinsic matrix to be relative to robot eef control frame
            assert cam_name.startswith("robot0") or cam_name.startswith("robot1")
            robot_ind = int(cam_name[5])
            if is_v15:
                eef_site_name = env.base_env.robots[robot_ind].composite_controller.part_controllers["right"].ref_name
            else:
                eef_site_name = env.base_env.robots[robot_ind].controller.eef_name
            eef_pos = np.array(env.base_env.sim.data.site_xpos[env.base_env.sim.model.site_name2id(eef_site_name)])
            eef_rot = np.array(env.base_env.sim.data.site_xmat[env.base_env.sim.model.site_name2id(eef_site_name)].reshape([3, 3]))
            eef_pose = np.zeros((4, 4)) # eef pose in world frame
            eef_pose[:3, :3] = eef_rot
            eef_pose[:3, 3] = eef_pos
            eef_pose[3, 3] = 1.0
            eef_pose_inv = np.zeros((4, 4))
            eef_pose_inv[:3, :3] = eef_pose[:3, :3].T
            eef_pose_inv[:3, 3] = -eef_pose_inv[:3, :3].dot(eef_pose[:3, 3])
            eef_pose_inv[3, 3] = 1.0
            R = R.dot(eef_pose_inv) # T_E^W * T_W^C = T_E^C
        camera_info[cam_name] = dict(
            intrinsics=K.tolist(),
            extrinsics=R.tolist(),
        )
    return camera_info



def make_env(args):
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.env_dataset_path)
    
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, 
        render=True, 
        render_offscreen=True,
        use_image_obs=True,
    )

    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)
    return env, is_robosuite_env


@dataclass
class PoseState:
    timestamp: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0

    def convert_frame_order(self):
        r_matrix = np.array([
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0]
        ])
        frame_rot = R.from_matrix(r_matrix)
        old_xyz = np.array([self.x, self.y, self.z])
        new_xyz = r_matrix @ old_xyz  
        old_rot = R.from_quat([self.qx, self.qy, self.qz, self.qw])
        new_rot = frame_rot * old_rot
        new_q = new_rot.as_quat()  
        return PoseState(timestamp=self.timestamp,x=new_xyz[0], y=new_xyz[1], z=new_xyz[2],qx=new_q[0], qy=new_q[1], qz=new_q[2], qw=new_q[3])
    

    def as_rotation(self):
        return R.from_quat([self.qx, self.qy, self.qz, self.qw])

    def __add__(self, other):
        combined_rot = self.as_rotation() * other.as_rotation()
        new_q = combined_rot.as_quat() # [x, y, z, w] 형태의 넘파이 배열 반환 (자동 정규화 됨)
        return PoseState(timestamp=0.0,x=self.x + other.x, y=self.y + other.y, z=self.z + other.z,qx=new_q[0], qy=new_q[1], qz=new_q[2], qw=new_q[3])

    def __sub__(self, other):
        diff_rot = self.as_rotation() * other.as_rotation().inv()
        new_q = diff_rot.as_quat()
        return PoseState(timestamp=0.0,x=self.x - other.x, y=self.y - other.y, z=self.z - other.z,qx=new_q[0], qy=new_q[1], qz=new_q[2], qw=new_q[3])

class AVP_handtracking():
        
    def __init__(self, udp_ip, udp_port, frame_id):
        self.udp_ip = udp_ip
        self.udp_port = udp_port 
        self.frame_id = frame_id
        self.is_grasp = False
        self.cur_phase_idx = 0

        print("initialize AVP handtracking instance")

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((udp_ip, udp_port))
        self.sock.setblocking(False)

        self.left_pose = PoseState()
        self.right_pose = PoseState()
        self.head_pose = PoseState()
        
        self.kb_listener = keyboard.Listener(on_press=self._on_keyboard_press)
        self.kb_listener.start()
        
        self.thread = threading.Thread(target=self.run, daemon=True)
        print("AVP handtracking thread starting...")
        self._is_running = True
        self.thread.start()

    def reset(self,):
        self.left_pose = PoseState()
        self.right_pose = PoseState()
        self.head_pose = PoseState()
        


    def _on_keyboard_press(self, key):
        try:
            if key.char == 'g':
                self.is_grasp = not self.is_grasp
            elif key.char == 'p':
                self.cur_phase_idx += 1
            elif key.char == 'q':
                self.close() # 기존에 만든 종료 메서드 호출
                return False
        except AttributeError:
            pass

    def get_current_phase(self,):
        return self.cur_phase_idx
    
    def get_current_gripper_state(self,):
        return self.is_grasp

    def get_current_tracker_poses(self,):
        # return self.left_pose, self.right_pose, self.head_pose
        return self.left_pose.convert_frame_order(), self.right_pose.convert_frame_order(), self.head_pose.convert_frame_order()
    

    def run(self):
        while self._is_running:
            self.poll_socket()
            time.sleep(0.005)

    def close(self):
        self._is_running = False
        try:
            self.sock.close()
        except Exception:
            pass
        if self.thread.is_alive():
            self.thread.join(timeout=1.0) # 스레드가 완전히 끝날 때까지 대기

    def __del__(self):
        self.close()

    def poll_socket(self):

        while True:
            try:
                data, addr = self.sock.recvfrom(4096)
            except BlockingIOError: break
            except Exception as e:
                self.get_logger().error(f"Socket error: {e}")
                break

            try: text = data.decode("utf-8").strip()
            except UnicodeDecodeError: continue

            parsed = self.parse_packet(text)
            if parsed is None: continue
            kind = parsed["kind"]
            if kind == "wrist_world":
                state = self.left_pose if parsed["chirality"] == "left" else self.right_pose
                qx, qy, qz, qw = self.normalized_quaternion(parsed["qx"], parsed["qy"], parsed["qz"], parsed["qw"])
                state.__dict__.update(timestamp=parsed["timestamp"], x=parsed["x"], y=parsed["y"], z=parsed["z"], qx=qx, qy=qy, qz=qz, qw=qw)

            elif kind == "head_world":
                qx, qy, qz, qw = self.normalized_quaternion(parsed["qx"], parsed["qy"], parsed["qz"], parsed["qw"])
                self.head_pose.__dict__.update(timestamp=parsed["timestamp"], x=parsed["x"], y=parsed["y"], z=parsed["z"], qx=qx, qy=qy, qz=qz, qw=qw)

    def parse_packet(self, text: str):
        """
        Supported UDP packet formats:

        wrist_world,timestamp,chirality,x,y,z,qx,qy,qz,qw
        head_world,timestamp,x,y,z,qx,qy,qz,qw
        gesture,timestamp,chirality,pinch,snap_left,snap_right,snap_up,snap_down,double_tap
        """
        parts = text.strip().split(",")

        if len(parts) == 10 and parts[0] == "wrist_world":
            try:
                _, timestamp, chirality, x, y, z, qx, qy, qz, qw = parts
                return {
                    "kind": "wrist_world",
                    "timestamp": float(timestamp),
                    "chirality": chirality.lower(),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "qx": float(qx),
                    "qy": float(qy),
                    "qz": float(qz),
                    "qw": float(qw),
                }
            except ValueError:
                return None

        if len(parts) == 9 and parts[0] == "head_world":
            try:
                _, timestamp, x, y, z, qx, qy, qz, qw = parts
                return {
                    "kind": "head_world",
                    "timestamp": float(timestamp),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "qx": float(qx),
                    "qy": float(qy),
                    "qz": float(qz),
                    "qw": float(qw),
                }
            except ValueError:
                return None


        return None


    def normalized_quaternion(self, x: float, y: float, z: float, w: float):
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-12:
            return 0.0, 0.0, 0.0, 1.0
        return x / norm, y / norm, z / norm, w / norm




def run_episode_avp(env, h_tracking, fps, initial_obs, is_dual = False, max_steps = 20 * 20): ## fps=20

    if is_dual: raise()


    traj = dict(
        obs=[], 
        next_obs=[], 
        rewards=[], 
        dones=[], 
        actions=[],
        states=[],
        phase_labels=[], 
    )

    obs = initial_obs
    
    while True:
        if h_tracking.get_current_tracker_poses()[1].timestamp > 0: break
        else: time.sleep(0.1)

    prev_pose = deepcopy(h_tracking.get_current_tracker_poses()[1]) # right-hand

    for t in range(max_steps):

        # if curr_pose.timestamp - prev_pose.timestamp > 1.0 / fps:
        while True:
            curr_pose = deepcopy(h_tracking.get_current_tracker_poses()[1])
            curr_gripper = deepcopy(h_tracking.get_current_gripper_state())
            curr_phase = deepcopy(h_tracking.get_current_phase())
            t_action = np.zeros(7)
            if curr_pose.timestamp - prev_pose.timestamp > 1.0 / fps: break
            time.sleep(0.5 * 1 / fps)
        
        # update
        delta_p =  curr_pose - prev_pose
        yaw, pitch, roll = R.from_quat([delta_p.qx, delta_p.qy, delta_p.qz, delta_p.qw]).as_euler('zyx', degrees=False)

        t_action[:3] = np.array([delta_p.x, delta_p.y, delta_p.z])
        t_action[3:6] = np.array([roll, pitch, yaw])
        t_action[:3] *= 35.0
        # t_action[3:6] *= 5.0
        t_action[5] *= 10.0  # yaw
        t_action[-1] = -1.0 if not curr_gripper else 1.0

        prev_pose = deepcopy(curr_pose)


        next_obs, r, done, _ = env.step(t_action)
        env.render(mode="human", height=512, width=512, camera_name="agentview")
        done = int(env.is_success()["task"])
        state = env.get_state()["states"]

        # collect transition
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)
        traj["phase_labels"].append(curr_phase)
        traj["states"].append(state)
        traj["actions"].append(t_action)

        obs = deepcopy(next_obs)
        time.sleep(1.0/fps)

        if done: break
        

    # fail
    # if not done: return None

    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else: traj[k] = np.array(traj[k])

    return traj


class KeyboardListenerThread:
    def __init__(self):
        # 1. 키보드 리스너 스레드 생성 및 시작
        # 주입된 on_press 함수가 키가 눌릴 때마다 백그라운드 스레드에서 자동 호출됩니다.
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        print("Keyboard input thread is running... (Press 'q' to quit)")

    def on_press(self, key):
        try:
            # 일반 문자 키('q', 'g', 'p' 등)의 값 추출
            char_key = key.char
            
            if char_key == 'q':
                print("\n['q'] 입력 감지: 프로그램을 종료합니다.")
                # 리스너 스레드를 안전하게 종료하려면 False를 리턴합니다.
                return False
                
            elif char_key == 'g':
                print("\n['g'] 입력 감지: 특정 액션(Go/Grab 등)을 수행합니다.")
                # 여기에 'g'가 눌렸을 때 실행할 로직이나 플래그 변경 작성
                
            elif char_key == 'p':
                print("\n['p'] 입력 감지: 일시정지(Pause) 또는 포즈 저장.")
                # 여기에 'p'가 눌렸을 때 실행할 로직 작성

        except AttributeError:
            # Shift, Ctrl, 방향키 등 특수 키가 눌렸을 때는 key.char가 없어 에러가 나므로 예외 처리
            pass

    def join(self):
        """리스너 스레드가 끝날 때까지 메인 스레드를 대기시키는 메서드"""
        self.listener.join()


def save_epi_hdf5(ep_data_grp, traj, initial_state, camera_info):
    ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
    ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
    ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
    ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
    ep_data_grp.create_dataset("phase_labels", data=np.array(traj["phase_labels"]))
    ep_data_grp.attrs["model_file"] = initial_state["model"]
    ep_data_grp.attrs["ep_meta"] = json.dumps(initial_state["ep_meta"])
    ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]
    ep_data_grp.attrs["camera_info"] = json.dumps(camera_info, indent=4)

    for k in traj["obs"]:
        ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="target path to hdf5 dataset")
    parser.add_argument("--env", type=str, required=True, choices=["square", "lift", "tool_hand", "can", "transport"], help="env / task name")
    parser.add_argument("--episodes", type=int, default=10, help="number of episodes for a hdf5 dataset")
    ############
    parser.add_argument("--interface", type=str, default="avp", choices=["keyboard", "avp"])
    parser.add_argument("--dual", action='store_true')
    parser.add_argument("--include-fail", action='store_true')
    parser.add_argument("--phase", action='store_true', help="save phase information")
    ############
    args = parser.parse_args()
    args.env_dataset_path = f"datasets/{args.env}/ph/image_v15.hdf5" 
    args.camera_names = ["agentview", "robot0_eye_in_hand"]
    args.fps = 20


    # make env
    env, is_robosuite_env = make_env(args)
    camera_info = get_camera_info(env, args.camera_names)
    print(json.dumps(env.serialize(), indent=4))
    print("is_robosuite_env: ", is_robosuite_env)
    np.set_printoptions(precision=5, suppress=True)



    # list of all demonstration episodes (sorted in increasing number order)
    f_in = h5py.File(args.env_dataset_path, "r")
    demos = list(f_in["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # output hdf5
    output_name = args.output
    output_path = os.path.join("/".join(args.env_dataset_path.split("/")[:-2]), args.interface, output_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")


    if args.interface == "keyboard":
        raise("Not implemented")
    else:
        udp_ip = "0.0.0.0"
        udp_port = 5005
        frame_id = "avp_world"

        h_tracking = AVP_handtracking(udp_ip, udp_port, frame_id)
        print("Use Apple Vision Pro to collect teleop demo")



    total_samples = 0
    for ep in range(args.episodes):  ## TODO

        h_tracking.reset()
        
        ## initialize env
        states = f_in[f"data/demo_{ep}/states"][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f_in[f"data/demo_{ep}"].attrs["model_file"]
            initial_state["ep_meta"] = f_in[f"data/demo_{ep}"].attrs.get("ep_meta", None)
        
        obs_modality_specs = {"obs": {"low_dim": ["robot0_eef_pos"], "rgb": ["{}_image".format(cn) for cn in args.camera_names],} }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=obs_modality_specs)
        initial_obs = env.reset_to(initial_state)

        ## run
        if args.interface == "avp":
            epi_traj = run_episode_avp(env, h_tracking, args.fps, initial_obs)

            if epi_traj is not None:
                ep_data_grp = data_grp.create_group(f"demo_{ep}")
                total_samples += epi_traj["actions"].shape[0]
                save_epi_hdf5(ep_data_grp, epi_traj, initial_state, camera_info)
                        
        else:
            # epi_traj = run_episode(env, initial_obs, ep)
            raise()

    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)

    f_in.close()
    f_out.close()