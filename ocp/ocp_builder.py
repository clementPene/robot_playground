import pinocchio as pin
import crocoddyl
import numpy as np

from ocp.cost_manager import CostModelManager
from ocp.contact_manager import ContactModelManager

class OCPBuilder:
    """Builds a Crocoddyl Optimal Control Problem (OCP).

    Attributes:
        rmodel (pin.Model): The Pinocchio model of the robot.
        initial_state (np.ndarray): The starting state (q, v) of the robot.
        dt (float): The time step duration for the integration scheme.
        horizon_length (int): The number of nodes (time steps) in the OCP horizon.
        state (crocoddyl.StateMultibody): The state model for the multibody system.
        actuation (crocoddyl.ActuationModelFloatingBase): The actuation model for a
            floating-base system, where the first 6 degrees of freedom are unactuated.
    """

    def __init__(self,
                 initial_state: np.ndarray,
                 rmodel: pin.Model,
                 dt: float,
                 horizon_length: int):
        """Initializes the OCPBuilder.

        Args:
            initial_state (np.ndarray): Initial state vector [q, v].
            rmodel (pin.Model): The Pinocchio robot model.
            dt (float): The time step (delta t) for each action model.
            horizon_length (int): The number of running nodes in the OCP. The total
                                  trajectory will have N+1 nodes.
        """
        self.rmodel = rmodel
        self.initial_state = initial_state
        self.dt = dt
        self.horizon_length = horizon_length

        self.state = crocoddyl.StateMultibody(self.rmodel) # input x = (q, v)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state) # output u = tau_q

    def set_initial_state(self, new_initial_state: np.ndarray):
        """Updates the initial state of the problem.

        Args:
            new_initial_state (np.ndarray): The new state vector [q, v].
        """
        if new_initial_state.shape[0] != self.state.nx:
            raise ValueError(f"The dimension of the new state ({new_initial_state.shape[0]}) "
                             f"does not match the expected dimension ({self.state.nx}).")
        self.initial_state = new_initial_state

    def build(self,
              running_cost_managers: list[CostModelManager],
              terminal_cost_manager: CostModelManager,
              running_contact_managers: list[ContactModelManager],
              terminal_contact_manager: ContactModelManager,
              integrator_type: str = 'euler') -> crocoddyl.ShootingProblem:
        """Constructs the Crocoddyl shooting problem.

        Args:
            running_cost_managers (List[CostModelManager]): A list of cost managers, one for
                each running node of the horizon.
            terminal_cost_manager (CostModelManager): The cost manager for the terminal node.
            running_contact_managers (List[ContactModelManager]): A list of contact managers, one
                for each running node, defining the contact sequence.
            terminal_contact_manager (ContactModelManager): The contact manager for the terminal node.
            integrator_type (str, optional): The integration scheme to use.
                Options: 'euler' or 'rk4'. Defaults to 'euler'.

        Returns:
            crocoddyl.ShootingProblem: The fully assembled optimal control problem.
        """        
        if len(running_cost_managers) != self.horizon_length:
            raise ValueError(f"number of 'running_cost_managers' ({len(running_cost_managers)}) "
                             f"must be equal to 'horizon_length' ({self.horizon_length}).")
        if len(running_contact_managers) != self.horizon_length:
            raise ValueError(f"Number of 'running_contact_managers' ({len(running_contact_managers)}) "
                             f"must be equal to 'horizon_length' ({self.horizon_length}).")

        running_models = self._create_running_models(running_cost_managers,
                                                     running_contact_managers,
                                                     integrator_type)
        terminal_model = self._create_terminal_model(terminal_cost_manager,
                                                     terminal_contact_manager,
                                                     integrator_type)

        problem = crocoddyl.ShootingProblem(self.initial_state, running_models, terminal_model)
        
        return problem


    def _create_running_models(self, 
                               cost_managers: list[CostModelManager], 
                               contact_managers: list[ContactModelManager],
                               integrator_type: str = 'euler') -> list:
        """Creates the list of integrated action models for the running nodes.

        Args:
            cost_managers (List[CostModelManager]): List of cost managers for the horizon.
            contact_managers (List[ContactModelManager]): List of contact managers for the horizon.
            integrator_type (str): The integration scheme ('euler' or 'rk4').

        Returns:
            List[crocoddyl.IntegratedActionModelAbstract]: The list of configured action models.
        """
        running_models = []
        for cost_manager, contact_manager in zip(cost_managers, contact_managers):
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, 
                self.actuation, 
                contact_manager.contact_model_sum,
                cost_manager.cost_model_sum
            )
            if integrator_type.lower() == 'euler':
                running_model = crocoddyl.IntegratedActionModelEuler(dmodel, self.dt)
            elif integrator_type.lower() == 'rk4':
                running_model = crocoddyl.IntegratedActionModelRK(dmodel, crocoddyl.RKType(4), self.dt)
            else:
                raise ValueError(f"Unknown integrator type: '{integrator_type}'. Choose 'euler' or 'rk4'.")

            running_models.append(running_model)
            
        return running_models


    def _create_terminal_model(self, 
                               cost_manager: CostModelManager, 
                               contact_manager: ContactModelManager,
                               integrator_type: str = 'euler') -> crocoddyl.IntegratedActionModelEuler:
        """Creates the integrated action model for the terminal node.

        Args:
            cost_manager (CostModelManager): The cost manager for the terminal node.
            contact_manager (ContactModelManager): The contact manager for the terminal node.
            integrator_type (str): The integration scheme ('euler' or 'rk4').

        Returns:
            crocoddyl.IntegratedActionModelAbstract: The configured terminal action model.
        """
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
            self.state, 
            self.actuation, 
            contact_manager.contact_model_sum, 
            cost_manager.cost_model_sum
        )

        if integrator_type.lower() == 'euler':
            terminal_model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.0)
        elif integrator_type.lower() == 'rk4':
            terminal_model = crocoddyl.IntegratedActionModelRK(dmodel, crocoddyl.RKType(4), 0.0)
        else:
            raise ValueError(f"Unknown integrator type: '{integrator_type}'. Choose 'euler' or 'rk4'.")

        return terminal_model
