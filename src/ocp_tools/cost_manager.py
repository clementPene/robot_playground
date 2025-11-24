import crocoddyl
import numpy as np
import pinocchio as pin
import yaml

class CostModelManager:
    """
    Cost structure to easily build a cost model with chainable cost functions
    """
    def __init__(self, state, actuation):
        self.state = state
        self.actuation = actuation
        self.cost_model_sum = crocoddyl.CostModelSum(self.state, self.actuation.nu)

    def add_regulation_state_cost(self, 
                                  x_ref: np.ndarray, 
                                  weight: float,
                                  name: str = "regulation_state"):
        """
        Add a state regulation cost.
        Isotropic weighting.

        This cost penalizes the difference between the current state `x` and a
        specified reference state `x_ref`.

        Mathematical Formulation:
        L(x) = (weight / 2) * ||x - x_ref||^2

        Args:
            x_ref (np.ndarray): The reference state vector [q_ref, v_ref] to track.
                            Its size must be equal to the state dimension (nx).
            weight (float): The scalar weight for this cost.
            name (str): A unique name for the cost.

        Returns:
            self: The CostModelManager instance for chainable calls.
        """

        # Input Validation
        if not isinstance(x_ref, np.ndarray):
            raise TypeError(f"x_ref must be a numpy array, but got {type(x_ref)}.")
        if x_ref.shape != (self.state.nx,):
            raise ValueError(
                f"The reference state x_ref must have shape ({self.state.nx},), "
                f"but got shape {x_ref.shape}."
            )
    
        residual = crocoddyl.ResidualModelState(self.state, x_ref, self.actuation.nu)
        cost = crocoddyl.CostModelResidual(self.state, residual)
        self.cost_model_sum.addCost(name=name, cost=cost, weight=weight)
        return self
    
    def add_weighted_regulation_state_cost(self,
                                           x_ref: np.ndarray,
                                           config_filepath: str,
                                           weight: float = 1.0,
                                           name: str = "regulation_weighted_state"):
        """Adds a weighted state regulation cost by loading weights from a YAML file.

        This cost penalizes the state deviation `x - x_ref` using an anisotropic quadratic function.
        The weights for each state variable are specified in the provided configuration file.

        Mathematical Formulation:

            The cost is defined as: L(x) = (weight / 2) * ||x ⊖ x_ref||_w^2
            Where:
            - `x` is the state vector.
            - `x_ref` is the reference state vector.
            - `w` is the vector of weights loaded from the YAML file.
            - `weight` is a global scalar weight for this cost term.

            Expanded Form:
            L(x) = weight * (1/2) * (w_1*(x_1 - x_ref_1)^2 + w_2*(x_2 - x_ref_2)^2 + ... + w_n*(x_n - x_ref_n)^2)

        Args:
            config_filepath (str): The path to the YAML file with the state weights. size is nv * 2
            x_ref (np.ndarray): The reference state vector [q_ref, v_ref] to track. size is nv + nq
            weight (float): A global scalar weight for the entire cost term.
            name (str): A unique name for the cost.

        Returns:
            self: The CostModelManager instance for chainable calls.
        """

        # Input Validation
        if not isinstance(x_ref, np.ndarray):
            raise TypeError(f"x_ref must be a numpy array, but got {type(x_ref)}.")
        if x_ref.shape != (self.state.nx,):
            raise ValueError(
                f"The reference state x_ref must have shape ({self.state.nx},), "
                f"but got shape {x_ref.shape}."
            )

        # Load and validate Weights from YAML file
        try:
            with open(config_filepath, 'r') as f:
                config_data = yaml.safe_load(f)

                q_weights = np.array(config_data['q_weights'])
                v_weights = np.array(config_data['v_weights'])

                weights = np.concatenate([q_weights, v_weights])

        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at: {config_filepath}")
        except KeyError as e:
            raise KeyError(f"Key {e} not found in the configuration file: {config_filepath}. Both 'q_weights' and 'v_weights' are required.")
        except Exception as e:
            raise RuntimeError(f"Failed to load or parse weights from {config_filepath}: {e}")

        # Check that the final weight vector matches the model's state dimension
        if len(weights) != 2 * self.state.nv:
            raise ValueError(
                f"Combined weights from file have size {len(weights)}, "
                f"but the model's state dimension is {2 * self.state.nv}."
            )

        # Create and add cost to the model
        residual = crocoddyl.ResidualModelState(self.state, x_ref, self.actuation.nu)
        activation = crocoddyl.ActivationModelWeightedQuad(weights)
        cost = crocoddyl.CostModelResidual(self.state, activation, residual)
        self.cost_model_sum.addCost(name=name, cost=cost, weight=weight)
        return self

    def add_regulation_control_cost(self,
                                    weight: float,
                                    u_ref: np.ndarray = None,
                                    name: str = "regulation_control"):
        """
        Add a command regulation cost.
        Isotropic weighting.

        This cost penalizes the difference between the current control `u` and a
        reference control `u_ref` (feedforward term). If `u_ref` is not provided,
        the cost defaults to penalizing the control effort `u`.

        Mathematical Formulation:
            L(u) = (weight / 2) * ||u - u_ref||^2

            Where:
            - `u` is the control vector.
            - `u_ref` is the reference control vector (defaults to a zero vector).
            - `weight` is a global scalar weight for this cost term.

        Args:
            weight (float): The scalar weight for this cost.
            u_ref (np.ndarray, optional): The reference control vector to track.
                                        Its size must be equal to the control dimension (nu).
                                        If None, a zero vector is used. Defaults to None.
            name (str): A unique name for the cost.

        Returns:
            self: The CostModelManager instance for chainable calls.
        """

        # Define and validate the reference control
        reference = u_ref if u_ref is not None else np.zeros(self.actuation.nu)
        if not isinstance(reference, np.ndarray):
            raise TypeError(f"u_ref must be a numpy array, but got {type(reference)}.")
        if reference.shape != (self.actuation.nu,):
            raise ValueError(
                f"The reference control u_ref must have shape ({self.actuation.nu},), "
                f"but got shape {reference.shape}."
            )

        residual = crocoddyl.ResidualModelControl(self.state, reference)
        cost = crocoddyl.CostModelResidual(self.state, residual)
        self.cost_model_sum.addCost(name=name, cost=cost, weight=weight)
        return self
    
    def add_weighted_regulation_control_cost(self,
                                             config_filepath: str,
                                             weight: float = 1.0,
                                             u_ref: np.ndarray = None,
                                             name: str = "regulation_weighted_control"):
        """
        Adds a weighted command regulation cost by loading weights from a YAML file.

        This cost penalizes the control input `u - u_ref` using an anisotropic quadratic function.
        The weights for each actuator are specified in the provided configuration file.

        Mathematical Formulation:
            L(u) = (weight / 2) * ||u - u_ref||_w^2
            
            Where:
            - `u` is the control vector.
            - `u_ref` is the reference control vector (defaults to a zero vector).
            - `w` is the vector of weights loaded from the YAML file.
            - `weight` is a global scalar weight for this cost term.

            Expanded Form:
            L(u) = (weight / 2) * sum_{i=1 to nu} [ w_i * (u_i - u_ref_i)^2 ]
        

        Args:
            config_filepath (str): The path to the YAML file containing the control weights.
                               The file must contain a key 'u_weights' with a list of numbers.
            weight (float): A global scalar weight for the entire cost term.
            u_ref (np.ndarray, optional): The reference control vector (feedforward).
                                        Must have size `nu`. Defaults to a zero vector if None.
            name (str): A unique name for the cost.

        Returns:
            self: The CostModelManager instance for chainable calls.
        """

        # Load and validate Weights from YAML file 
        try:
            with open(config_filepath, 'r') as f:
                config_data = yaml.safe_load(f)
            
                weights_list = config_data['u_weights']
                weights = np.array(weights_list)

        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at: {config_filepath}")
        except KeyError:
            raise KeyError(f"'u_weights' key not found in the configuration file: {config_filepath}")
        except Exception as e:
            raise RuntimeError(f"Failed to load or parse weights from {config_filepath}: {e}")

        if weights.ndim != 1 or len(weights) != self.actuation.nu:
            raise ValueError(
                f"Loaded 'u_weights' must be a 1D array of size {self.actuation.nu}, "
                f"but got size {len(weights)}."
            )
        
        # Define and validate the reference control
        reference = u_ref if u_ref is not None else np.zeros(self.actuation.nu)
        if not isinstance(reference, np.ndarray):
            raise TypeError(f"u_ref must be a numpy array, but got {type(reference)}.")
        if reference.shape != (self.actuation.nu,):
            raise ValueError(
                f"The reference control u_ref must have shape ({self.actuation.nu},), "
                f"but got shape {reference.shape}."
            )
    
        # Create and add cost to the model
        residual = crocoddyl.ResidualModelControl(self.state, reference)
        activation = crocoddyl.ActivationModelWeightedQuad(weights)
        cost = crocoddyl.CostModelResidual(self.state, activation, residual)
        self.cost_model_sum.addCost(name=name, cost=cost, weight=weight)

        return self
    
    # TODO -> need the model
    def add_state_limits_cost(self,
                              weight: float,
                              name: str = "state_limits"):
        """
        Adds a cost to enforce state (position and velocity) limits.

        This is a crucial safety cost that prevents the optimizer from generating
        trajectories that are physically impossible or damaging for the robot hardware.
        It penalizes any joint position or velocity that goes beyond the limits
        specified in the robot's URDF file (and loaded by Pinocchio).

        Mathematical Formulation:
            The cost penalizes any component of the state vector `x` that violates its bounds.
            Let `l` be the lower_bounds and `u` be the upper_bounds. The cost is zero
            if `l <= x <= u` (component-wise). Otherwise, it grows quadratically.
            The bounds for the floating base are set to infinity, as it is unconstrained.

        Args:
            weight (float): The scalar weight for this cost. High values are recommended
                            to treat these limits as hard constraints.
            name (str): A unique name for the cost.

        Returns:
            self: The CostModelManager instance for chainable calls.
        """
        # Input Validation
        if not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError(f"weight must be a non-negative number, but got {weight}.")

        # --- Define Bounds from Pinocchio Model ---
        # The state vector x is [q, v].
        # q = [base_pose (7D), joint_positions]
        # v = [base_velocity (6D), joint_velocities]

        # Lower bounds
        lower_q = np.concatenate([
            -np.inf * np.ones(6),                 # Floating base pose (unbounded in tangent space)
            self.rmodel.lowerPositionLimit[7:]   # Joint position limits
        ])
        lower_v = -self.rmodel.velocityLimit     # Velocity limits (symmetric)
        lower_bounds = np.concatenate([lower_q, lower_v])

        # Upper bounds
        upper_q = np.concatenate([
            np.inf * np.ones(6),                  # Floating base pose
            self.rmodel.upperPositionLimit[7:]    # Joint position limits
        ])
        upper_v = self.rmodel.velocityLimit      # Velocity limits
        upper_bounds = np.concatenate([upper_q, upper_v])

        # Crocoddyl's state residual is applied on the tangent space, which has size nx = nv+nv.
        # We must therefore provide bounds of size nx.
        bounds = crocoddyl.ActivationBounds(lower_bounds, upper_bounds)
        activation = crocoddyl.ActivationModelQuadraticBarrier(bounds)

        # The residual is the state itself. The activation will penalize it if it's out of bounds.
        residual = crocoddyl.ResidualModelState(self.state, self.actuation.nu)
        cost = crocoddyl.CostModelResidual(self.state, activation, residual)

        self.cost_model_sum.addCost(name=name, cost=cost, weight=weight)
        return self
    
    # TODO -> need the model
    def add_control_limits_cost(self,
                                weight: float,
                                name: str = "control_limits"):
        """
        Adds a cost to enforce control (motor torque/effort) limits.

        This is a critical physical realism cost. It ensures that the commands
        sent to the motors are within the torque limits they can actually produce,
        preventing the solver from finding solutions that are not executable.
        The limits are retrieved from the robot's URDF file (loaded by Pinocchio).

        Mathematical Formulation:
            The cost penalizes any component of the control vector `u` that violates its bounds.
            Let `l` be the lower_bounds and `u` be the upper_bounds. The cost is zero
            if `l <= u <= u` (component-wise). Otherwise, it grows quadratically.

        Args:
            weight (float): The scalar weight for this cost. High values are recommended
                            to treat these limits as hard constraints.
            name (str): A unique name for the cost.

        Returns:
            self: The CostModelManager instance for chainable calls.
        """
        # Input Validation
        if not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError(f"weight must be a non-negative number, but got {weight}.")

        # Define Bounds from Pinocchio Model
        # Control u corresponds to actuated joints. For a floating-base robot,
        # we skip the first 6 virtual joints.
        lower_bounds = -self.rmodel.effortLimit[6:]
        upper_bounds = self.rmodel.effortLimit[6:]

        bounds = crocoddyl.ActivationBounds(lower_bounds, upper_bounds)
        activation = crocoddyl.ActivationModelQuadraticBarrier(bounds)

        # The residual is the control vector itself.
        residual = crocoddyl.ResidualModelControl(self.state, self.actuation.nu)
        cost = crocoddyl.CostModelResidual(self.state, activation, residual)

        self.cost_model_sum.addCost(name=name, cost=cost, weight=weight)
        return self
    
    def add_contact_friction_cone_cost(self,
                                       contact_ids: list[int],
                                       mu: float,
                                   weight: float,
                                   R_cone: np.ndarray = None,
                                   name: str = "contact_friction"):
        """
        Adds a friction cone cost for one or more contacts.

        This cost ensures that the contact forces remain within the friction cone,
        which is a fundamental constraint for physically realistic contact interactions.
        It penalizes forces that would cause slipping.

        Mathematical Formulation:
            The cost penalizes any force vector `f` that lies outside the friction cone.
            A friction cone is defined by the inequality:
                sqrt(f_x^2 + f_y^2) <= mu * f_z
            where `f_z` is the normal force and `f_x`, `f_y` are the tangential forces.
            The cost is zero inside the cone and grows quadratically outside.

        Args:
            contact_ids (list[int]): A list of frame IDs for the contacts to be constrained.
            mu (float): The coefficient of friction (typically between 0.1 and 1.0).
            weight (float): The scalar weight for this cost.
            R_cone (np.ndarray, optional): A 3x3 rotation matrix for the surface normal.
                                        The Z-axis of this matrix defines the normal direction.
                                        Defaults to np.eye(3) for a flat, horizontal ground.
            name (str): A base name for the cost. The frame ID will be appended to it
                        to create a unique name for each contact cost.

        Returns:
            self: The CostModelManager instance for chainable calls.
        """
        # --- Input Validation ---
        if not isinstance(contact_ids, (list, tuple)):
            raise TypeError(f"contact_ids must be a list or tuple of integers, but got {type(contact_ids)}.")
        if not all(isinstance(i, int) for i in contact_ids):
            raise TypeError("All elements in contact_ids must be integers.")
        if not mu > 0:
            raise ValueError(f"Friction coefficient mu must be positive, but got {mu}.")

        rotation_cone = R_cone if R_cone is not None else np.eye(3)
        if not isinstance(rotation_cone, np.ndarray) or rotation_cone.shape != (3, 3):
            raise ValueError(f"R_cone must be a 3x3 numpy array, but got shape {rotation_cone.shape}.")

        # Cost Creation
        friction_cone = crocoddyl.FrictionCone(rotation_cone, mu)

        # Loop through each contact ID and add a dedicated cost
        for contact_id in contact_ids:
            residual = crocoddyl.ResidualModelContactFrictionCone(
                self.state,
                contact_id,
                friction_cone,
                self.actuation.nu
            )
            cost = crocoddyl.CostModelResidual(self.state, residual)
            
            # Generate a unique and descriptive name for each cost
            unique_name = f"{name}_{contact_id}"
            self.cost_model_sum.addCost(name=unique_name, cost=cost, weight=weight)
            
        return self
    
    def add_contact_force_bounds_cost(self,
                                  contact_ids: list[int],
                                  lower_bounds: np.ndarray,
                                  upper_bounds: np.ndarray,
                                  weight: float,
                                  name: str = "contact_force_bounds"):
        """
        Adds a cost to penalize contact forces that go outside a predefined range.

        This cost defines a "safe zone" for the 6D contact forces (linear and angular).
        It is more flexible than tracking a single reference force, making it ideal for
        dynamic movements where forces naturally vary. It prevents physically unrealistic
        scenarios, such as the foot pulling on the ground (negative normal force) or
        excessive torques that would cause the foot to tilt.

        Mathematical Formulation:
            The cost penalizes any component of the force vector `f` that violates its bounds.
            Let `l` be the lower_bounds and `u` be the upper_bounds. The cost is zero
            if `l <= f <= u` (component-wise). Otherwise, it grows quadratically.
            For any component `f_i` of the force vector:
                - cost(f_i) = 0                  if l_i <= f_i <= u_i
                - cost(f_i) = 0.5 * (f_i - u_i)^2 if f_i > u_i
                - cost(f_i) = 0.5 * (l_i - f_i)^2 if f_i < l_i
            This is achieved using an `ActivationModelQuadraticBarrier` in Crocoddyl.

        Args:
            contact_ids (list[int]): A list of frame IDs for the contacts to be constrained.
            lower_bounds (np.ndarray): A 6D numpy array `[fx_min, fy_min, fz_min, tx_min, ty_min, tz_min]`
                                       representing the minimum allowed forces and torques.
            upper_bounds (np.ndarray): A 6D numpy array `[fx_max, fy_max, fz_max, tx_max, ty_max, tz_max]`
                                       representing the maximum allowed forces and torques.
            weight (float): The scalar weight for this cost.
            name (str): A base name for the cost. The frame ID will be appended to it
                        to create a unique name for each contact cost.

        Returns:
            self: The CostModelManager instance for chainable calls.
        """
        # Input Validation
        if not isinstance(contact_ids, list):
            raise TypeError(f"contact_ids must be a list, but got {type(contact_ids)}.")
        if not isinstance(lower_bounds, np.ndarray) or lower_bounds.shape != (6,):
            raise ValueError(f"lower_bounds must be a 6D numpy array, but got shape {lower_bounds.shape}.")
        if not isinstance(upper_bounds, np.ndarray) or upper_bounds.shape != (6,):
            raise ValueError(f"upper_bounds must be a 6D numpy array, but got shape {upper_bounds.shape}.")
        if not np.all(lower_bounds <= upper_bounds):
            raise ValueError("All elements of lower_bounds must be <= to their counterparts in upper_bounds.")

        # Cost creation
        bounds = crocoddyl.ActivationBounds(lower_bounds, upper_bounds)

        activation = crocoddyl.ActivationModelQuadraticBarrier(bounds, 1.0)
        
        for contact_id in contact_ids:
            residual = crocoddyl.ResidualModelContactForce(
                self.state,
                contact_id,
                pin.Force.Zero(), # ref_force is ignored by the activation, but required
                6,
                self.actuation.nu
            )

            # Create the cost model with our specific barrier activation
            cost = crocoddyl.CostModelResidual(self.state, activation, residual)
            
            unique_name = f"{name}_{contact_id}"
            self.cost_model_sum.addCost(name=unique_name, cost=cost, weight=weight)
            
        return self

    def add_CoM_position_cost(self,
                              com_position_ref: np.ndarray,
                              weight: float,
                              name: str = "com_position"):
        """
        Adds a cost to track a reference Center of Mass (CoM) position.

        This cost is fundamental for controlling the robot's balance and overall
        movement. It penalizes the squared distance between the robot's current
        Center of Mass position and a desired target position in 3D space.

        Mathematical Formulation:
        L(x) = (weight / 2) * || CoM(q) - com_position_ref ||^2
        where CoM(q) is the position of the center of mass computed from the
        configuration q of the state vector x.

        Args:
            com_position_ref (np.ndarray): The target 3D position [x, y, z] for the
                                           Center of Mass. Must be a numpy array
                                           of shape (3,).
            weight (float): The scalar weight for this cost.
            name (str): A unique name for the cost.

        Returns:
            self: The CostModelManager instance for chainable calls.
        """
        # --- Input Validation ---
        if not isinstance(com_position_ref, np.ndarray):
            raise TypeError(
                f"com_position_ref must be a numpy array, but got {type(com_position_ref)}."
            )
        if com_position_ref.shape != (3,):
            raise ValueError(
                f"The reference CoM position must have shape (3,), "
                f"but got shape {com_position_ref.shape}."
            )
        if not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError(f"weight must be a non-negative number, but got {weight}.")

        # Create cost
        residual = crocoddyl.ResidualModelCoMPosition(
            self.state, com_position_ref, self.actuation.nu
        )
        
        cost = crocoddyl.CostModelResidual(self.state, residual)

        self.cost_model_sum.addCost(name=name, cost=cost, weight=weight)
        
        return self

    def add_state_corridor_cost(self,
                                x_ref: np.ndarray,
                                config_filepath: str,
                                weight: float = 1.0,
                                scale: float = 1.0,
                                name: str = "state_corridor"):
        """
        Add a “corridor” cost around the reference state using a quadratic barrier
        on the state residual r = diff(x_ref, x).

        Inside the bounds (|r_i| <= Δ_i): zero cost.
        Outside: quadratic penalty on the violation.

        Args:
            x_ref : np.ndarray
                Reference state of size nx = nq + nv (here 34 + 33 = 67 for your model).
            config_filepath : str
                Path to a YAML file defining the corridor widths (Δ) for dq and dv.
            weight : float
                Global weight applied to this cost term (scales the barrier).
            scale : float
                Homotopy scaling factor to tighten/loosen all bounds at once: Δ <- scale * Δ.
            name : str
                Unique name for this term inside the CostModelSum.
            
        Returns:
            self: The CostModelManager instance for chainable calls.
        """

        if not isinstance(x_ref, np.ndarray):
            raise TypeError(f"x_ref must be a numpy array, but got {type(x_ref)}.")

        if x_ref.shape != (self.state.nx,):
            raise ValueError(
                f"x_ref must have shape ({self.state.nx},), but got {x_ref.shape}."
            )

        try:
            with open(config_filepath, "r") as f:
                cfg = yaml.safe_load(f)

            q_tube = np.asarray(cfg["q_tube"], dtype=float)  # size is nv
            v_tube = np.asarray(cfg["v_tube"], dtype=float)  # size is nv
            yaml_scale = float(cfg.get("scale", 1.0))
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at: {config_filepath}")
        except KeyError as e:
            raise KeyError(
                f"Key {e} missing in YAML {config_filepath}. "
                f"Expected keys: 'q_tube' and 'v_tube' (both of length nv={self.state.nv})."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load/parse YAML {config_filepath}: {e}")

        if q_tube.shape != (self.state.nv,) or v_tube.shape != (self.state.nv,):
            raise ValueError(
                f"'q_tube' and 'v_tube' must both have shape ({self.state.nv},). "
                f"Got q_tube={q_tube.shape}, v_tube={v_tube.shape}."
            )
        if np.any(q_tube < 0.0) or np.any(v_tube < 0.0):
            raise ValueError("All corridor widths (q_tube, v_tube) must be non-negative.")

        # Build bounds on the residual r (dimension ndx)
        delta = np.concatenate([q_tube, v_tube]) * (scale * yaml_scale)  # Δ >= 0
        if delta.shape != (self.state.ndx,):
            raise RuntimeError(f"Internal error: delta has shape {delta.shape}, expected ({self.state.ndx},).")

        lb = -delta
        ub = +delta

        # Quadratic barrier activation on r
        bounds = crocoddyl.ActivationBounds(lb, ub)
        activation = crocoddyl.ActivationModelQuadraticBarrier(bounds)

        # State residual in tangent space (r = diff(x_ref, x))
        residual = crocoddyl.ResidualModelState(self.state, x_ref, self.actuation.nu)
        
        cost = crocoddyl.CostModelResidual(self.state, activation, residual)
        self.cost_model_sum.addCost(name=name, cost=cost, weight=weight)
        return self

    def get_costs(self):
        """
        Return the final constructed CostModelManager object.

        Returns:
            crocoddyl.CostModelManager: The set of configured costs.
        """
        return self.cost_model_sum
    
    def display_costs(self):
        """
        Prints a detailed, human-readable summary of all costs configured
        in this cost model.
        """
        print("\n---- Cost Model Summary ----")
        
        costs_map = self.cost_model_sum.costs
        
        if not costs_map:
            print("  No costs have been configured in this model.")
            print("--------------------------\n")
            return
        
        print(f"  Total number of costs: {len(costs_map)}")
        print("--------------------------")
        
        for item in costs_map:
            # a. Extraire le nom via l'attribut .key
            name = item.key
            
            # b. Extraire l'objet CostItem en appelant la méthode .data()
            cost_item = item.data()
            
            # c. Extraire les informations et les afficher de manière formatée
            weight = cost_item.weight
            cost_type = type(cost_item.cost).__name__ # Utilise .__name__ pour un affichage plus propre
            
            print(f"  > Cost Name: '{name}'")
            print(f"    - Weight: {weight}")
            print(f"    - Type  : {cost_type}")
        
        print("--------------------------\n")
        
        
