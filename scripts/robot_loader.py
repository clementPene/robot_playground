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
    
    urdf_filename = os.path.join(project_dir, "robot_models/biped_assets/biped_assets/models/h12/h12_12dof.urdf")
    mesh_dir = os.path.join(project_dir, "robot_models/biped_assets/biped_assets/models/h12")

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
            mesh_path
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

# This part only runs if you launch "python robot_loader.py"
if __name__ == '__main__':
    print("Testing the path loading module...")

    urdf_path, mesh_path = load_robot_paths()
    
    if urdf_path:
        print("\nModule is working !")
        print(f"URDF path found: {urdf_path}")
        print(f"Mesh path found: {mesh_path}")
        print("\nTest successful !")


