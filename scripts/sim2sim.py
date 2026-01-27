import time
import numpy as np
import pinocchio as pin
import crocoddyl

from robot_deploy.simulators import MujocoSim

from ocp_tools.ocp_builder import OCPBuilder
from ocp_tools.cost_manager import CostModelManager
from ocp_tools.contact_manager import ContactModelManager

class MPCController:
    def __init__(self, pin_model, pin_data, q0):
        self.model = pin_model
        self.data = pin_data

        # --------- paramètres OCP ---------
        self.OCP_DT = 0.02        # sera le dt de contrôle MuJoCo aussi
        self.HORIZON_LENGTH = 50
        self.TRAJECTORY_DURATION = 3.0
        self.N_OCP_STEPS = int(self.TRAJECTORY_DURATION / self.OCP_DT)

        # --------- état initial ----------
        v0 = np.zeros(self.model.nv)
        a0 = np.zeros(self.model.nv)
        full_gravity_torques = pin.rnea(self.model, self.data, q0, v0, a0)
        self.u0 = full_gravity_torques[6:]         # torques joints
        self.x0 = np.concatenate([q0, v0])

        # --------- trajectoire CoM --------
        self.reference_com_trajectory = self._build_com_trajectory(q0)

        # --------- OCP builder ------------
        self.ocp_build = OCPBuilder(
            initial_state=self.x0,
            rmodel=self.model,
            dt=self.OCP_DT,
            horizon_length=self.HORIZON_LENGTH,
        )

        # --------- contacts ---------------
        (self.running_contact_managers,
         self.terminal_contact_manager) = self._build_contacts()

        # --------- coûts ------------------
        conf_weights = 'config/H1_squat/regulation_state_weights.yaml'
        (self.running_cost_managers,
         self.terminal_cost_manager,
         self.running_com_costs) = self._build_costs(conf_weights)

        # --------- problème + solver ------
        self.problem = self.ocp_build.build(
            self.running_cost_managers,
            self.terminal_cost_manager,
            self.running_contact_managers,
            self.terminal_contact_manager,
        )
        self.solver = crocoddyl.SolverFDDP(self.problem)
        self.MAX_ITER = 50  # réduire pour tenir en temps réel

        # warm start
        self.xs_init = [self.x0 for _ in range(self.HORIZON_LENGTH + 1)]
        self.us_init = [self.u0 for _ in range(self.HORIZON_LENGTH)]

        # index courant dans la trajectoire
        self.i = 0

    # ===================== PUBLIC ===================== #
    def step(self, x_current):
        """
        x_current : vecteur [q, v] courant (format Pinocchio / OCP)
        Retourne : (dt, torques)
        """
        # état initial du problème = état courant
        self.problem.x0 = x_current.copy()

        # mise à jour des références de CoM sur l’horizon
        current_com_ref = []
        for j in range(self.HORIZON_LENGTH):
            idx = self.i + j
            idx = min(idx, len(self.reference_com_trajectory) - 1)
            ref = self.reference_com_trajectory[idx]
            self.running_com_costs[j].residual.reference = ref
            current_com_ref.append(ref.copy())

        # résoudre l’OCP (MPC)
        converged = self.solver.solve(self.xs_init, self.us_init, self.MAX_ITER, False)
        if not converged:
            print(f"[WARN] FDDP {self.i} n’a pas convergé (iter={self.solver.iter}, cost={self.solver.cost})")

        # première commande de l’horizon = ce qu’on applique
        u_optimal = self.solver.us[0].copy()

        # mise à jour du warm start
        self.xs_init = list(self.solver.xs)[1:] + [self.solver.xs[-1]]
        self.us_init = list(self.solver.us)[1:] + [self.solver.us[-1]]

        self.i += 1

        return self.OCP_DT, u_optimal

    # ===================== PRIVÉ : helpers ===================== #
    def _build_com_trajectory(self, q0):
        # --- ton code existant ici, adapté ---
        pin.forwardKinematics(self.model, self.data, q0, np.zeros(self.model.nv))
        pin.updateFramePlacements(self.model, self.data)
        pin.centerOfMass(self.model, self.data, q0, np.zeros(self.model.nv))

        z_com_highest = self.data.com[0][2].copy()
        z_com_lower = z_com_highest - 0.20

        mean_com = (z_com_highest + z_com_lower) / 2.0
        amplitude = (z_com_highest - z_com_lower) / 2.0

        N = self.N_OCP_STEPS
        z_traj = [mean_com + amplitude * np.cos(2 * np.pi * t / N) for t in range(N)]
        z_traj.extend([z_com_highest] * self.HORIZON_LENGTH)

        ref_traj = [np.array([0.011, 0.001, z]) for z in z_traj]
        return ref_traj

    def _build_contacts(self):
        running_contact_managers = []
        for _ in range(self.HORIZON_LENGTH):
            cm = ContactModelManager(self.ocp_build.state,
                                     self.ocp_build.actuation,
                                     self.model, self.data)
            cm.add_contact_6D("left_ground").add_contact_6D("right_ground")
            running_contact_managers.append(cm)

        terminal_cm = ContactModelManager(self.ocp_build.state,
                                          self.ocp_build.actuation,
                                          self.model, self.data) \
            .add_contact_6D("left_ground") \
            .add_contact_6D("right_ground")
        return running_contact_managers, terminal_cm

    def _build_costs(self, config_filepath):
        running_cost_managers = []
        running_com_costs = []

        for j in range(self.HORIZON_LENGTH):
            cm = CostModelManager(self.ocp_build.state, self.ocp_build.actuation)
            cm.add_CoM_position_cost(np.array(self.reference_com_trajectory[j]), 1e2) \
              .add_weighted_regulation_state_cost(x_ref=self.x0,
                                                  config_filepath=config_filepath,
                                                  weight=1e-2) \
              .add_regulation_control_cost(1e-6)
            # .add_contact_friction_cone_cost(contact_frame_ids, 0.7, 1e-6)

            running_cost_managers.append(cm)
            running_com_costs.append(cm.differential.costs.costs['com_position'].cost)

        terminal_cm = CostModelManager(self.ocp_build.state, self.ocp_build.actuation)
        terminal_cm.add_regulation_state_cost(x_ref=self.x0, weight=1e-2)

        return running_cost_managers, terminal_cm, running_com_costs

