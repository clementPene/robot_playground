import numpy as np

def build_pinocchio_state(raw_state_vector, source_indices):
    """
    Parses a raw state vector from the trajectory file and converts it
    into Pinocchio-compliant q and v vectors. 
    This is done because the raw state vector describing the trajectory 
    does not match the Pinocchio model's joint order.
    """
    # 1. Extract components from the raw vector
    base_pos = raw_state_vector[0:3]
    base_quat_wxyz = raw_state_vector[3:7] # Format [w, x, y, z]
    base_lin_vel = raw_state_vector[7:10]
    base_ang_vel = raw_state_vector[10:13]
    joint_pos_raw = raw_state_vector[13:40]
    joint_vel_raw = raw_state_vector[40:67]

    # 2. Re-format for Pinocchio
    
    # Permute quaternion to [x, y, z, w]
    pinocchio_quat = np.array([base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]])
    
    # Reorder joint data using the pre-calculated index map
    reordered_joint_pos = joint_pos_raw[source_indices]
    reordered_joint_vel = joint_vel_raw[source_indices]

    # 3. Assemble the final q and v vectors
    q = np.concatenate([base_pos, pinocchio_quat, reordered_joint_pos])
    v = np.concatenate([base_lin_vel, base_ang_vel, reordered_joint_vel])
    
    return q, v

# --- Usage Example ---
# Assuming robot model is loaded
# q, v = build_pinocchio_state(trajectory_data[i], source_indices)
# viz.display(q)
