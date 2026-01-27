from dataclasses import dataclass
import crocoddyl
import numpy as np

@dataclass
class MPCOCP:
    problem: crocoddyl.ShootingProblem
    solver: crocoddyl.SolverFDDP
    dt: float
    horizon_length: int


class MPCController:
    def __init__(
        self,
        ocp: MPCOCP,
        x0: np.ndarray,
        u0: np.ndarray | None = None,
        max_iter: int = 5,
        com_ref_traj: np.ndarray | None = None,
        verbose: bool = True,
    ):
        """
        ocp : instance de MPCOCP (problem + solver + dt + horizon_length)
        """
        self.ocp = ocp
        self.dt = ocp.dt
        self.N = ocp.horizon_length
        
        self.com_ref_traj = com_ref_traj
        self.running_com_costs = [
            m.differential.costs.costs["com_position"].cost
            for m in self.ocp.problem.runningModels
        ]

        # Compteur d'itérations MPC
        self.k = 0
        self.max_iter = max_iter
        self.verbose = verbose

        # Dimensions
        model0 = ocp.problem.runningModels[0]
        self.nx = model0.state.nx
        self.nu = model0.nu
        
        if u0 is None:
            u0 = np.zeros(self.nu)
        
        self.xs_init = [x0.copy() for _ in range(self.N + 1)]
        self.us_init = [u0.copy() for _ in range(self.N)]
        
        self.ocp.problem.x0 = x0.copy()
        
        
    def _update_initial_state(self, x_meas: np.ndarray):
        """
            Update the initial state of the OCP with the measured state. 
            Need to be called before each OCP solve.
            Also update warm start initial state.
        """
        self.xs_init[0] = x_meas.copy()
        self.ocp.problem.x0 = x_meas.copy()
        
    def update_reference_com(self):
        """
            This function will be specific to the task and will update the CoM references
            in the cost models of the OCP.
            TODO : find a better way to do this ?
        """
        if self.com_ref_traj is None:
            return

        T = len(self.com_ref_traj)

        for j in range(self.N):
            idx = self.k + j
            if idx >= T:
                idx = T - 1  # hold last position

            ref = self.com_ref_traj[idx]
            self.running_com_costs[j].residual.reference = ref

    
    def reset(self, x0: np.ndarray, u0: np.ndarray | None = None):
        self.k = 0
        if u0 is None:
            u0 = np.zeros(self.nu)
        self.xs_init = [x0.copy() for _ in range(self.N + 1)]
        self.us_init = [u0.copy() for _ in range(self.N)]
        self.ocp.problem.x0 = x0.copy()

    def step(self, x_meas: np.ndarray, max_iter: int | None = None):
        """
        Un pas de MPC :
          - met à jour l'état initial avec x_meas,
          - résout l'OCP,
          - renvoie (dt, u0).
        """
        
        self._update_initial_state(x_meas)
        self.update_reference_com()
        
        if max_iter is None:
            max_iter = self.max_iter
            
        converged = self.ocp.solver.solve(
            self.xs_init,
            self.us_init,
            max_iter,
            False,
        )
        
        #logger
        if self.verbose:
            if converged:
                print(f"SUCCESS: FDDP {self.k} converged in {self.ocp.solver.iter} iterations.")
            else:
                print(f"WARNING: FDDP {self.k} did not converge (iter={self.ocp.solver.iter}, cost={self.ocp.solver.cost}).")
        
        # get result from solver to be used as next input
        u_optimal = self.ocp.solver.us[0].copy()

        # Update warm start
        xs = self.ocp.solver.xs
        us = self.ocp.solver.us
        self.xs_init = [xs[i].copy() for i in range(1, len(xs))]
        self.xs_init.append(xs[-1].copy())

        self.us_init = [us[i].copy() for i in range(1, len(us))] 
        self.us_init.append(us[-1].copy())  

        self.k += 1
        return self.dt, u_optimal