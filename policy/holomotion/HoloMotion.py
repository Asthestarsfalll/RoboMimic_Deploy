from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.utils import FSMCommand, get_gravity_orientation
import numpy as np
import yaml
import os
import json
import h5py
import onnxruntime
import onnx
from collections import deque


class HoloMotion(FSMState):
    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.SKILL_HOLOMOTION
        self.name_str = "holomotion"
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "HoloMotion.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
        # Model paths
        self.onnx_path = os.path.join(current_dir, "model", config["onnx_path"])
        self.hdf5_root = config["hdf5_root"]
        
        # Control parameters
        self.kps = np.array(config["kps"], dtype=np.float32)
        self.kds = np.array(config["kds"], dtype=np.float32)
        self.default_angles = np.array(config["default_angles"], dtype=np.float32)
        self.joint2motor_idx = np.array(config["joint2motor_idx"], dtype=np.int32)
        self.tau_limit = np.array(config["tau_limit"], dtype=np.float32)
        self.num_actions = config["num_actions"]
        self.num_obs = config["num_obs"]
        self.action_scale = np.array(config["action_scale"], dtype=np.float32)
        self.context_length = config["context_length"]
        
        # DOF names and mappings
        self.dof_names_onnx = config["dof_names_onnx"]
        self.default_angles_onnx = np.array(config["default_angles_onnx"], dtype=np.float32)
        
        # Initialize buffers
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        
        # Initialize observation history buffers using deque
        # Each buffer stores [context_length, feat_dim]
        # Store (queue, feat_dim) tuples for each buffer
        self.obs_buffers = {
            "ref_motion_states": (deque(maxlen=self.context_length), 2 * self.num_actions),  # 58
            "projected_gravity": (deque(maxlen=self.context_length), 3),  # 3
            "rel_robot_root_ang_vel": (deque(maxlen=self.context_length), 3),  # 3
            "dof_pos": (deque(maxlen=self.context_length), self.num_actions),  # 29
            "dof_vel": (deque(maxlen=self.context_length), self.num_actions),  # 29
            "last_action": (deque(maxlen=self.context_length), self.num_actions),  # 29
        }
        # Track number of pushes for warm start
        self.obs_buffer_pushes = {key: 0 for key in self.obs_buffers.keys()}
        
        # Load ONNX model
        self.ort_session = onnxruntime.InferenceSession(self.onnx_path)
        input = self.ort_session.get_inputs()
        self.input_name = []
        for i, inpt in enumerate(input):
            self.input_name.append(inpt.name)
        
        # Load HDF5 motion data
        self._load_motion_data()
        
        # Motion state
        self.motion_frame_idx = 0
        self.current_motion_idx = 0
        
        print("HoloMotion policy initializing ...")
    
    def _load_motion_data(self):
        """Load motion data from HDF5 dataset."""
        manifest_path = os.path.join(self.hdf5_root, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        self.dof_names_ref_motion = manifest["dof_names"]
  
        name_mapping = {
            "torso_yaw_joint": "waist_yaw_joint",
            "torso_roll_joint": "waist_roll_joint",
            "torso_pitch_joint": "waist_pitch_joint",
        }
        
        self.ref_to_onnx = []
        missing_names = []
        for name in self.dof_names_onnx:
            # Try direct match first
            if name in self.dof_names_ref_motion:
                self.ref_to_onnx.append(self.dof_names_ref_motion.index(name))
            # Try mapped name (torso -> waist)
            elif name in name_mapping and name_mapping[name] in self.dof_names_ref_motion:
                self.ref_to_onnx.append(self.dof_names_ref_motion.index(name_mapping[name]))
            else:
                missing_names.append(name)
        
        if missing_names:
            raise ValueError(
                f"DOF names not found in reference motion: {missing_names}\n"
                f"ONNX names ({len(self.dof_names_onnx)}): {self.dof_names_onnx}\n"
                f"HDF5 names ({len(self.dof_names_ref_motion)}): {self.dof_names_ref_motion}\n"
                f"Available HDF5 DOF names: {self.dof_names_ref_motion}"
            )
        
        self.motion_clips = list(manifest.get("clips", {}).keys())
        if len(self.motion_clips) == 0:
            raise ValueError("No motion clips found in manifest")
        
        # Load first motion clip as example
        self._load_motion_clip(self.motion_clips[0])
    
    def _load_motion_clip(self, motion_key: str):
        """Load a specific motion clip from HDF5."""
        manifest_path = os.path.join(self.hdf5_root, "manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        clip_info = manifest["clips"][motion_key]
        shard_idx = clip_info["shard"]
        start = clip_info["start"]
        length = clip_info["length"]
        
        shard_file = os.path.join(
            self.hdf5_root, 
            manifest["hdf5_shards"][shard_idx]["file"]
        )
        
        with h5py.File(shard_file, "r") as f:
            self.ref_dof_pos = f["dof_pos"][start:start+length].astype(np.float32)
            self.ref_dof_vel = f["dof_vels"][start:start+length].astype(np.float32)
        
        self.n_motion_frames = length
        self.motion_frame_idx = 0
    
    def _compute_projected_gravity(self, quat):
        """Compute gravity vector projected into robot's root frame.
        
        Args:
            quat: Quaternion [w, x, y, z]
        
        Returns:
            Projected gravity vector [x, y, z] in root frame
        """
        # Use the existing get_gravity_orientation function
        # It expects [w, x, y, z] format
        return get_gravity_orientation(quat).astype(np.float32)
    
    def _build_observation(self, q, dq, base_quat, ang_vel, ref_dof_pos, ref_dof_vel):
        """Build observation from current state and reference motion.
        
        Args:
            q: Current joint positions [29]
            dq: Current joint velocities [29]
            base_quat: Base quaternion [w, x, y, z]
            ang_vel: Angular velocity [3]
            ref_dof_pos: Reference joint positions [29] (in ref motion order)
            ref_dof_vel: Reference joint velocities [29] (in ref motion order)
        
        Returns:
            Flattened observation vector [604]
        """
        # Convert base_quat to [w, x, y, z] format if needed
        if len(base_quat) == 4:
            imu_quat = base_quat
        else:
            imu_quat = np.array([base_quat[3], base_quat[0], base_quat[1], base_quat[2]], dtype=np.float32)
        
        # Map reference motion to ONNX order
        ref_dof_pos_onnx = ref_dof_pos[self.joint2motor_idx].astype(np.float32)
        ref_dof_vel_onnx = ref_dof_vel[self.joint2motor_idx].astype(np.float32)
        
        # Map current joint states to ONNX order
        # Assuming q and dq are already in the correct order (MuJoCo order)
        # We need to map them to ONNX order if needed
        # For now, assume they match the joint2motor_idx mapping
        q_onnx = np.zeros(self.num_actions, dtype=np.float32)
        dq_onnx = np.zeros(self.num_actions, dtype=np.float32)
        for i, motor_idx in enumerate(self.joint2motor_idx):
            if i < len(q) and motor_idx < len(q):
                q_onnx[i] = q[motor_idx] - self.default_angles_onnx[i]
                dq_onnx[i] = dq[motor_idx]
        
        ref_motion_states = np.concatenate([ref_dof_pos_onnx, ref_dof_vel_onnx], axis=0).astype(np.float32)
        projected_gravity = self._compute_projected_gravity(imu_quat)
        rel_robot_root_ang_vel = ang_vel.astype(np.float32)
        dof_pos = q_onnx.astype(np.float32)
        dof_vel = dq_onnx.astype(np.float32)
        last_action = self.last_action.astype(np.float32)
        buffer_data = {
            "ref_motion_states": ref_motion_states,
            "projected_gravity": projected_gravity,
            "rel_robot_root_ang_vel": rel_robot_root_ang_vel,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "last_action": last_action,
        }
        
        for key, data in buffer_data.items():
            queue, feat_dim = self.obs_buffers[key]
            if self.obs_buffer_pushes[key] == 0:
                # Warm start: duplicate first push across entire history
                for _ in range(self.context_length):
                    queue.append(data.copy())
            else:
                queue.append(data.copy())
            self.obs_buffer_pushes[key] += 1
        
        obs_list = []
        for key in ["ref_motion_states", "projected_gravity", "rel_robot_root_ang_vel", 
                    "dof_pos", "dof_vel", "last_action"]:
            queue, feat_dim = self.obs_buffers[key]
            if len(queue) == 0:
                hist = np.zeros((self.context_length, feat_dim), dtype=np.float32)
            else:
                hist = np.array(list(queue), dtype=np.float32)
            obs_list.append(hist.reshape(-1))
        
        hist_obs = np.concatenate(obs_list, axis=0).astype(np.float32)
        return hist_obs
    
    def enter(self):
        """Initialize state when entering."""
        self.motion_frame_idx = 0
        # Reset all observation buffers
        for key in self.obs_buffers.keys():
            queue, _ = self.obs_buffers[key]
            queue.clear()
            self.obs_buffer_pushes[key] = 0
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        
        # Load a random motion clip
        if len(self.motion_clips) > 0:
            import random
            motion_key = random.choice(self.motion_clips)
            self._load_motion_clip(motion_key)
    
    def run(self):
        """Execute policy inference."""
        # Get robot state
        q = self.state_cmd.q.copy()
        dq = self.state_cmd.dq.copy()
        base_quat = self.state_cmd.base_quat.copy()  # [w, x, y, z]
        ang_vel = self.state_cmd.ang_vel.copy()
        
        # Get reference motion state
        frame_idx = min(self.motion_frame_idx, self.n_motion_frames - 1)
        ref_dof_pos_raw = self.ref_dof_pos[frame_idx]
        ref_dof_vel_raw = self.ref_dof_vel[frame_idx]
        
        # Build observation
        hist_obs = self._build_observation(
            q=q,
            dq=dq,
            base_quat=base_quat,
            ang_vel=ang_vel,
            ref_dof_pos=ref_dof_pos_raw,
            ref_dof_vel=ref_dof_vel_raw,
        )
        
        # Run policy inference
        observation = {self.input_name[0]: hist_obs.reshape(1, -1).astype(np.float32)}
        outputs = self.ort_session.run(None, observation)
        self.action = outputs[0].squeeze()
        
        # Convert action to motor indices
        target_dof_pos = self.action * self.action_scale + self.default_angles
        target_dof_pos_mj = np.zeros(29)
        target_dof_pos_mj[self.joint2motor_idx] = target_dof_pos
        
        self.policy_output.actions = target_dof_pos_mj
        self.policy_output.kps[self.joint2motor_idx] = self.kps
        self.policy_output.kds[self.joint2motor_idx] = self.kds
        
        # Update state
        self.last_action = self.action.copy()
        self.motion_frame_idx = (self.motion_frame_idx + 1) % self.n_motion_frames
    
    def exit(self):
        """Clean up when exiting."""
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.motion_frame_idx = 0
        # Reset buffers
        for key in self.obs_buffers.keys():
            queue, _ = self.obs_buffers[key]
            queue.clear()
            self.obs_buffer_pushes[key] = 0
        print("HoloMotion policy exited")
    
    def checkChange(self):
        """Check if state should change."""
        if self.state_cmd.skill_cmd == FSMCommand.LOCO:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.LOCOMODE
        elif self.state_cmd.skill_cmd == FSMCommand.PASSIVE:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        elif self.state_cmd.skill_cmd == FSMCommand.POS_RESET:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.FIXEDPOSE
        else:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_HOLOMOTION
