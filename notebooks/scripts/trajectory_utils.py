import numpy as np


# Joint order as defined in the Pinocchio URDF model
_PINOCCHIO_JOINT_ORDER = [
    'left_hip_yaw_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 
    'right_hip_yaw_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 
    'torso_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 
    'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 
    'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 
    'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 
    'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
]

# Joint order as found in the source trajectory file
_TRAJECTORY_FILE_ORDER = [
    'left_hip_yaw_joint', 'right_hip_yaw_joint', 'torso_joint', 
    'left_hip_pitch_joint', 'right_hip_pitch_joint', 'left_shoulder_pitch_joint', 
    'right_shoulder_pitch_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 
    'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 
    'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 
    'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_joint', 
    'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 
    'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 
    'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint'
]

# Indices for reordering (calculated only once on module import)
_SOURCE_INDICES = [_TRAJECTORY_FILE_ORDER.index(joint) for joint in _PINOCCHIO_JOINT_ORDER]

_INVERSE_JOINT_INDICES = np.empty_like(_SOURCE_INDICES)
_INVERSE_JOINT_INDICES[_SOURCE_INDICES] = np.arange(len(_TRAJECTORY_FILE_ORDER))

def convert_trajectory_to_pinocchio_format(raw_trajectory_data):
    """
    Reorders a raw trajectory and separates it into configuration (q) and 
    velocity (v) trajectories, ready for Pinocchio algorithms.

    The function handles the quaternion permutation (from wxyz to xyzw) and
    the reordering of joint position/velocity columns.

    Args:
        raw_trajectory_data (np.ndarray): 
            The raw trajectory array of shape (N, 67).

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            A tuple containing two arrays:
                - q_trajectory (np.ndarray): Trajectory of configurations, shape (N, nq).
                - v_trajectory (np.ndarray): Trajectory of velocities, shape (N, nv).
    """
    # Ensure the trajectory is 2D (N, 67)
    if raw_trajectory_data.ndim == 3:
        raw_trajectory_data = np.squeeze(raw_trajectory_data, axis=1)

    # 1. Slice the raw trajectory
    base_pos = raw_trajectory_data[:, 0:3]
    base_quat_wxyz = raw_trajectory_data[:, 3:7]
    base_lin_vel = raw_trajectory_data[:, 7:10]
    base_ang_vel = raw_trajectory_data[:, 10:13]
    joint_pos_raw = raw_trajectory_data[:, 13:40]
    joint_vel_raw = raw_trajectory_data[:, 40:67]

    # 2. Reorder the data
    pinocchio_quat_xyzw = base_quat_wxyz[:, [1, 2, 3, 0]]
    reordered_joint_pos = joint_pos_raw[:, _SOURCE_INDICES]
    reordered_joint_vel = joint_vel_raw[:, _SOURCE_INDICES]

    # 3. Re-assemble into separate vectors
    q_trajectory = np.concatenate([
        base_pos,
        pinocchio_quat_xyzw,
        reordered_joint_pos,
    ], axis=1)

    v_trajectory = np.concatenate([
        base_lin_vel,
        base_ang_vel,
        reordered_joint_vel
    ], axis=1)

    print(f"raw trajectory dimension (raw_trajectory.shape): {raw_trajectory_data.shape}")
    print(f"q trajectory dimension (q_trajectory.shape): {q_trajectory.shape}")
    print(f"v trajectory dimension (v_trajectory.shape): {v_trajectory.shape}")

    print("Raw trajectory converted and separated into q and v trajectories.")
    return q_trajectory, v_trajectory


def convert_pinocchio_to_trajectory_format(q_trajectory, v_trajectory):
    """
    Reconstructs the raw trajectory format from Pinocchio-formatted 
    configuration (q) and velocity (v) trajectories.

    This function is the inverse of 'convert_trajectory_to_pinocchio_format'.
    It handles the inverse quaternion permutation (from xyzw to wxyz) and
    the reordering of joint position/velocity columns back to their original order.

    Args:
        q_trajectory (np.ndarray): 
            Trajectory of configurations, Pinocchio format, shape (N, 34).
        v_trajectory (np.ndarray): 
            Trajectory of velocities, Pinocchio format, shape (N, 33).

    Returns:
        np.ndarray: 
            The reconstructed raw trajectory array, shape (N, 67).
    """
    # Slice the q and v trajectories
    base_pos = q_trajectory[:, :3]
    pinocchio_quat_xyzw = q_trajectory[:, 3:7]
    pinocchio_joint_pos = q_trajectory[:, 7:]
    
    base_lin_vel = v_trajectory[:, :3]
    base_ang_vel = v_trajectory[:, 3:6]
    pinocchio_joint_vel = v_trajectory[:, 6:]

    # Reverse the data reordering
    # Invert the quaternion permutation from xyzw (Pinocchio) to wxyz (source)
    base_quat_wxyz = pinocchio_quat_xyzw[:, [3, 0, 1, 2]]

    # Invert the joint reordering to return to the file order
    joint_pos_raw = pinocchio_joint_pos[:, _INVERSE_JOINT_INDICES]
    joint_vel_raw = pinocchio_joint_vel[:, _INVERSE_JOINT_INDICES]

    # Re-assemble into a single raw trajectory vector
    raw_trajectory_data = np.concatenate([
        base_pos,
        base_quat_wxyz,
        base_lin_vel,
        base_ang_vel,
        joint_pos_raw,
        joint_vel_raw
    ], axis=1)

    print(f"q trajectory dimension (q_trajectory.shape): {q_trajectory.shape}")
    print(f"v trajectory dimension (v_trajectory.shape): {v_trajectory.shape}")
    print(f"Reconstituted raw trajectory dimension (raw_trajectory.shape): {raw_trajectory_data.shape}")

    print("Pinocchio q and v trajectories converted back to raw trajectory format.")
    return raw_trajectory_data
