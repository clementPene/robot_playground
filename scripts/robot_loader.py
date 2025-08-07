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

# This part only runs if you launch "python robot_loader.py"
if __name__ == '__main__':
    print("Testing the path loading module...")

    urdf_path, mesh_path = load_robot_paths()
    
    if urdf_path:
        print("\nModule is working !")
        print(f"URDF path found: {urdf_path}")
        print(f"Mesh path found: {mesh_path}")
        print("\nTest successful !")


