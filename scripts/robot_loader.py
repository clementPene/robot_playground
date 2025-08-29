import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import os
import numpy as np
import meshcat.geometry as g

def load_robot_paths():
    """
    This function handles paths to load the robot

    Returns:
        tuple: A tuple containing (urdf_path, mesh_dir)
    """
    print("--- Searching for robot paths ---")

    current_file_path = os.path.abspath(__file__)
    current_file_dir = os.path.dirname(current_file_path)
    project_dir = os.path.dirname(current_file_dir)

    print(f"Detected project root directory: {project_dir}")
    
    urdf_filename = os.path.join(project_dir, "robot_models/h1_2_description/h1_2_handless.urdf")
    mesh_dir = os.path.join(project_dir, "robot_models/h1_2_description")

    assert os.path.exists(urdf_filename), f"URDF file not found: {urdf_filename}"
    assert os.path.exists(mesh_dir), f"Mesh directory not found: {mesh_dir}"
    print("Asset paths verified.")

    print(f"Detected urdf file: {urdf_filename}")
    print(f"Detected mesh directory: {mesh_dir}")

    return urdf_filename, mesh_dir

def launch_visualization():
    """
    Load and configure the robot, initialize Meshcat and set up the visualizer.

    This function encapsulates the following steps:
    1. Loading the robot paths.
    2. Launching the Meshcat visualizer.
    3. Loading the robot model into Pinocchio.
    4. Initializing the Pinocchio visualizer.

    Returns:
        tuple: A tuple containing the ready-to-use objects:
               (model, collision_model, visual_model, robot_visualizer, viz)
               Returns (None, None, None, None, None) in case of error.
    """

    try:
        urdf_path, mesh_path = load_robot_paths()
    except AssertionError as e:
        print(f"Error in path files : {e}")
        return None, None, None, None, None

    print("Launching MeshCat...")
    viz = meshcat.Visualizer()
    viz.open()
    print("Adding a grid to represent the ground.")
    viz["/Grid"].set_property("visible", True)

    print("Loading robot model into Pinocchio...")
    try:
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            urdf_path,
            mesh_path,
            pin.JointModelFreeFlyer()
        )
        print("Pinocchio model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None, viz # Return viz to see the grid even if the robot fails
    
    robot_visualizer = MeshcatVisualizer(model, collision_model, visual_model)
    robot_visualizer.initViewer(viewer=viz)
    robot_visualizer.loadViewerModel()
    print("\nEnvironment is ready !")

    return model, collision_model, visual_model, robot_visualizer, viz


def add_contact_frames(model, 
                       left_foot_parent_frame_name='left_ankle_roll_link', 
                       right_foot_parent_frame_name='right_ankle_roll_link', 
                       contact_z_offset=0.05):
    """
    Adds operational frames under the feet to represent ground contact points.

    This function modifies the Pinocchio model by adding two new frames,
    'left_ground_contact' and 'right_ground_contact'. These frames are positioned
    relative to their parent ankle frames with a vertical offset.

    Args:
        model (pin.Model): The Pinocchio model to be modified.
        left_foot_parent_frame_name (str): The name of the existing frame in the left foot 
                                           to which the new contact frame will be attached.
        right_foot_parent_frame_name (str): The name of the existing frame in the right foot.
        contact_z_offset (float): The vertical distance to translate downwards from the 
                                  parent frame to the new contact frame.

    Returns:
        pin.Model: The modified model with the new contact frames.
    """

    left_contact_frame_name = "left_ground_contact"
    right_contact_frame_name = "right_ground_contact"

    # Create the SE(3) transformation for the offset
    # This represents a pure translation along the parent frame's Z-axis.
    offset_translation = np.array([0., 0., -contact_z_offset])
    contact_offset_pose = pin.SE3(np.eye(3), offset_translation)

    # Get parent frame information
    try:
        left_foot_parent_id = model.getFrameId(left_foot_parent_frame_name)
        right_foot_parent_id = model.getFrameId(right_foot_parent_frame_name)
    except KeyError as e:
        print(f"Error: A parent frame could not be found in the model: {e}")
        print("Aborting frame creation.")
        return model

    # Add the left contact frame if it does not already exist
    if not model.existFrame(left_contact_frame_name):
        model.addFrame(pin.Frame(left_contact_frame_name,
                                 model.frames[left_foot_parent_id].parentJoint,
                                 model.frames[left_foot_parent_id].placement * contact_offset_pose,
                                 pin.FrameType.OP_FRAME))
        print(f"Frame '{left_contact_frame_name}' added successfully.")
    else:
        print(f"Frame '{left_contact_frame_name}' already exists. Skipping.")

    # Add the right contact frame if it does not already exist
    if not model.existFrame(right_contact_frame_name):
        model.addFrame(pin.Frame(right_contact_frame_name,
                                 model.frames[right_foot_parent_id].parentJoint,
                                 model.frames[right_foot_parent_id].placement * contact_offset_pose,
                                 pin.FrameType.OP_FRAME))
        print(f"Frame '{right_contact_frame_name}' added successfully.")
    else:
        print(f"Frame '{right_contact_frame_name}' already exists. Skipping.")

    # --- this part is to thune CONTACT_Z_OFFSET ---
    #sphere_geometry1 = mg.Sphere(0.04)
    #left_ankle_pose = data.oMf[left_foot_middle_frame_id]
    #sphere_pose = left_ankle_pose * contact_offset_pose
    #red_material = mg.MeshLambertMaterial(color=0xff0000)
    #viz[f"markers/{left_contact_frame_name}"].set_object(sphere_geometry1, red_material)
    #viz[f"markers/{left_contact_frame_name}"].set_transform(sphere_pose.homogeneous)

    #sphere_geometry2 = mg.Sphere(0.04)
    #viz[f"markers/{right_foot_frame_id}"].set_object(sphere_geometry2, red_material)
    #viz[f"markers/{right_foot_frame_id}"].set_transform(right_frame_pose.homogeneous)

    return model

# This part only runs if you launch "python robot_loader.py"
if __name__ == '__main__':
    print("Testing the path loading module...")

    urdf_path, mesh_path = load_robot_paths()
    
    if urdf_path:
        print("\nModule is working !")
        print(f"URDF path found: {urdf_path}")
        print(f"Mesh path found: {mesh_path}")
        print("\nTest successful !")


