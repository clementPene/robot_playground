import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import os
import numpy as np
import meshcat.geometry as g

def load_robot_and_visualizer():
    """
    This function handles the entire process of loading the H12 robot
    and configuring the MeshCat visualizer.

    It returns the essential objects for interacting with the robot.

    Returns:
        tuple: A tuple containing (model, viz, robot_visualizer)
    """
    print("--- Starting load process via robot_loader module ---")

    # --- 1. Configure Paths ---
    # We use a path relative to this file, which makes the code more portable.
    current_file_path = os.path.abspath(__file__)
    current_file_dir = os.path.dirname(current_file_path)
    project_dir = os.path.dirname(current_file_dir)

    print(f"Detected project root directory: {project_dir}")
    
    urdf_filename = os.path.join(project_dir, "robot_models/biped_assets/biped_assets/models/h12/h12_12dof.urdf")
    mesh_dir = os.path.join(project_dir, "robot_models/biped_assets/biped_assets/models/h12")

    assert os.path.exists(urdf_filename), f"URDF file not found: {urdf_filename}"
    assert os.path.exists(mesh_dir), f"Mesh directory not found: {mesh_dir}"
    print("Asset paths verified.")

    # --- 2. Launch and connect to MeshCat visualizer ---
    print("Launching MeshCat...")
    viz = meshcat.Visualizer()
    print("Adding a grid to represent the ground.")
    viz["/Grid"].set_property("visible", True)

    # --- 3. Load robot into Pinocchio ---
    print("Loading robot model into Pinocchio...")
    try:
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            urdf_filename,
            mesh_dir,
            pin.JointModelFreeFlyer()
        )
        print("Pinocchio model loaded successfully.")
    except Exception as e:
        print(f"Error loading robot: {e}")
        return None, None, None

    # --- 4. Create Pinocchio visualizer ---
    robot_visualizer = MeshcatVisualizer(model, collision_model, visual_model)

    print("--- Loading complete. Objects are ready to be used. ---")

    # Return the objects that will be useful in the notebook
    return model, viz, robot_visualizer

# This part only runs if you launch "python robot_loader.py"
# It's perfect for testing the module independently
if __name__ == '__main__':
    print("Testing the loading module...")

    model, viz, robot_visualizer = load_robot_and_visualizer()
    
    if model:
        print("\nThe module worked! Initializing test visualization.")

        # Open the MeshCat window if it's not already open
        viz.open()

        # Initialize the visualizer with the models
        robot_visualizer.initViewer(viewer=viz)
        robot_visualizer.loadViewerModel()

        # Display the robot in a raised neutral position
        q0 = pin.neutral(model)
        q0[2] = 1.03 # With this value, it seems that the robot feets are close to the ground
        robot_visualizer.display(q0)

        print("\nTest successful! A robot should be visible in MeshCat.")
        print("Press Enter to quit.")
        input()
