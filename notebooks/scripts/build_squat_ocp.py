import numpy as np
import crocoddyl
import pinocchio as pin

from ocp_tools.ocp_builder import OCPBuilder
from ocp_tools.cost_manager import CostModelManager
from ocp_tools.contact_manager import ContactModelManager
from mpc_controller.mpc_builder import MPCOCP


def build_squat_ocp(x0: np.ndarray,
                    model: pin.Model,
                    data: pin.Data,
                    reference_com_trajectory: list[np.ndarray],
                    contact_frame_ids: list[int],
                    dt: float,
                    horizon_length: int) -> MPCOCP:
    """
    This function is the result of experimentation made on page 3 and 4.
    It build the OCP problem that will be used for the MPC.

    Returns:
        MPCOCP: A class containing all the OCP informations :
            - problem
            - solver
            - dt
            - horizon_length
    """
    # initialize OCP
    ocp_build = OCPBuilder(initial_state=x0,
                                    rmodel=model,
                                    dt=dt,
                                    horizon_length=horizon_length)
    
    # Create contacts
    running_contact_managers = []
    for j in range (horizon_length):
        contact_manager = ContactModelManager(ocp_build.state, ocp_build.actuation, model, data)
        contact_manager.add_contact_6D("left_ground") \
                    .add_contact_6D("right_ground")
        running_contact_managers.append(contact_manager)

    terminal_contact_manager = ContactModelManager(ocp_build.state, ocp_build.actuation, model, data) \
                .add_contact_6D("left_ground") \
                .add_contact_6D("right_ground")
                
    # Create costs
    running_cost_managers = []
    config_filepath = 'config/H1_squat/regulation_state_weights.yaml'
    for j in range (horizon_length):
        cost_manager = CostModelManager(ocp_build.state, ocp_build.actuation)
        cost_manager.add_CoM_position_cost(np.array(reference_com_trajectory[j]), 1e2) \
                    .add_weighted_regulation_state_cost(x_ref=x0, config_filepath=config_filepath, weight=1e-2) \
                    .add_regulation_control_cost(1e-6) \
                    .add_contact_friction_cone_cost(contact_frame_ids, 0.7, 1e-6)

        running_cost_managers.append(cost_manager)

    terminal_cost_manager = CostModelManager(ocp_build.state, ocp_build.actuation)
    terminal_cost_manager.add_regulation_state_cost(x_ref=x0, weight=1e-2)
   
    # Finalize OCP
    problem = ocp_build.build(running_cost_managers, terminal_cost_manager, running_contact_managers, terminal_contact_manager)
    solver = crocoddyl.SolverFDDP(problem)
    

    return MPCOCP(
        problem=problem,
        solver=solver,
        dt=dt,
        horizon_length=horizon_length,
    )

def build_squat_com_trajectory(
    model,
    data,
    q0,
    v0,
    traj_duration: float,
    ocp_dt: float,
    horizon_length: int,
    squat_depth: float = 0.20,
) -> np.ndarray:
    """
    Construit une trajectoire de CoM pour un squat :
      - départ en position initiale,
      - descente puis remontée en 'traj_duration' secondes (cosinus),
      - puis maintien en position finale pendant 'horizon_length' pas de plus.

    Retourne un tableau de forme (T_total, 3) avec (x, y, z) du CoM à chaque pas OCP.
    """

    pin.forwardKinematics(model, data, q0, v0)
    pin.updateFramePlacements(model, data)
    pin.centerOfMass(model, data, q0, v0)

    com0 = data.com[0].copy()
    x_com_init, y_com_init, z_com_highest = com0

    z_com_lower = z_com_highest - squat_depth

    mean_com = 0.5 * (z_com_highest + z_com_lower)
    amplitude = 0.5 * (z_com_highest - z_com_lower)

    n_traj_steps = int(traj_duration / ocp_dt)

    t = np.linspace(0.0, traj_duration, n_traj_steps, endpoint=False)

    z_traj = mean_com + amplitude * np.cos(2.0 * np.pi * t / traj_duration)

    z_extension = np.full(horizon_length, z_traj[-1])
    z_all = np.concatenate([z_traj, z_extension])

    x_all = np.full_like(z_all, x_com_init)
    y_all = np.full_like(z_all, y_com_init)

    reference_com_trajectory = np.column_stack([x_all, y_all, z_all])
    return reference_com_trajectory