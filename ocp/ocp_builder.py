import pinocchio
import crocoddyl
import numpy as np

from ocp.cost_manager import CostModelManager


class OCPBuilder:

    def __init__(self,
                 initial_state: np.ndarray,
                 robot_model: pinocchio.Model,
                 dt: float,
                 horizon_length: int):
        """
        Initialise le constructeur de problème.

        Args:
            initial_state (np.ndarray): L'état initial du robot [q0, v0].
            robot_model (pinocchio.Model): Le modèle Pinocchio du robot.
            dt (float): Le pas de temps pour l'intégration (en secondes).
            horizon_length (int): Le nombre de nœuds dans la trajectoire (l'horizon).
        """
        # --- Validation des entrées ---
        if not isinstance(robot_model, pinocchio.Model):
            raise TypeError("robot_model must be an instance of pinocchio.Model.")
        if not isinstance(initial_state, np.ndarray) or initial_state.shape[0] != robot_model.nq + robot_model.nv:
            raise ValueError(f"initial_state must be a numpy array of shape ({robot_model.nq + robot_model.nv},).")
        
        self.robot_model = robot_model
        self.initial_state = initial_state
        self.dt = dt
        self.horizon_length = horizon_length

        # --- Initialisation des modèles de base de Crocoddyl ---
        # Le modèle d'état représente l'espace de configuration du robot
        self.state = crocoddyl.StateMultibody(self.robot_model)
        
        # Le modèle d'actuation décrit les forces que le robot peut appliquer
        self.actuation = crocoddyl.ActuationModelFull(self.state)
        
        print("ProblemBuilder initialisé.")


    def build(self,
              running_cost_manager: CostModelManager,
              terminal_cost_manager: CostModelManager) -> crocoddyl.ShootingProblem:
        """
        Assemble et retourne le ShootingProblem complet.

        Cette méthode est le point d'entrée principal. Elle crée la séquence
        de modèles d'action en utilisant les gestionnaires de coûts fournis.

        Args:
            running_cost_manager (CostModelManager): Un gestionnaire contenant
                tous les coûts qui s'appliquent à chaque nœud de la trajectoire
                (de 0 à N-1).
            terminal_cost_manager (CostModelManager): Un gestionnaire contenant
                les coûts qui s'appliquent UNIQUEMENT au nœud final (N).

        Returns:
            crocoddyl.ShootingProblem: L'objet problème, prêt à être utilisé par un solveur.
        """
        print("Construction du problème d'optimisation...")

        # --- Création des modèles d'action ---
        # "Running models" sont les modèles pour tous les pas de temps sauf le dernier.
        running_models = self._create_running_models(running_cost_manager)
        
        # "Terminal model" est le modèle spécifique pour le dernier pas de temps.
        terminal_model = self._create_terminal_model(terminal_cost_manager)

        # --- Assemblage du problème ---
        # Un ShootingProblem est défini par un état initial et une séquence de modèles d'action.
        problem = crocoddyl.ShootingProblem(self.initial_state, running_models, terminal_model)
        
        print("Problème construit avec succès !")
        return problem


    def _create_running_models(self, cost_manager: CostModelManager) -> list:
        """Crée la liste des modèles d'action pour les nœuds intermédiaires."""
        
        # Le modèle d'action différentielle combine la dynamique du robot avec
        # la somme des coûts.
        # `DifferentialActionModelFreeFwdDynamics` est le choix standard pour les
        # robots à base flottante.
        dmodel = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, cost_manager.cost_model_sum
        )

        # Le modèle différentiel est ensuite intégré numériquement dans le temps
        # en utilisant un schéma comme celui d'Euler.
        running_model = crocoddyl.IntegratedActionModelEuler(dmodel, self.dt)

        # On retourne une liste contenant le même modèle dupliqué N fois.
        return [running_model] * self.horizon_length


    def _create_terminal_model(self, cost_manager: CostModelManager) -> crocoddyl.IntegratedActionModelEuler:
        """Crée le modèle d'action pour le nœud terminal."""

        # Le modèle terminal est structurellement similaire, mais utilise le
        # `cost_model_sum` du gestionnaire de coûts terminaux.
        # Typiquement, il n'y a pas de coût de commande ici.
        dmodel = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, cost_manager.cost_model_sum
        )

        # Même s'il n'y a pas de "pas de temps" suivant, Crocoddyl attend
        # un modèle intégré pour le nœud terminal.
        terminal_model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.0) # dt=0.0 car c'est le dernier noeud

        return terminal_model



if __name__ == '__main__':
    # Cette section montre comment vous utiliseriez la classe.
    
    # 1. Charger un modèle de robot (exemple avec un modèle simple)
    #    Dans votre cas, vous chargerez votre robot URDF.
    robot_model = pinocchio.buildSampleModelHumanoid()
    robot_data = robot_model.createData()

    # 2. Définir les paramètres du problème
    q0 = robot_model.referenceConfigurations['neutral']
    v0 = np.zeros(robot_model.nv)
    x0 = np.concatenate([q0, v0])
    
    TIME_STEP = 0.01  # 10 ms
    HORIZON_LENGTH = 100 # 1 seconde de trajectoire

    # 3. Initialiser le constructeur de problème
    problem_builder = ProblemBuilder(initial_state=x0,
                                     robot_model=robot_model,
                                     dt=TIME_STEP,
                                     horizon_length=HORIZON_LENGTH)
                                     
    # 4. Configurer les coûts pour les noeuds courants (running costs)
    running_costs = CostModelManager(robot_model, robot_data)
    running_costs.add_regulation_state_cost(x_ref=x0, weight=0.1, name="state_reg")
    # running_costs.add_control_limits_cost(weight=1.0)
    # running_costs.add_CoM_position_cost(...)

    # 5. Configurer les coûts pour le noeud final (terminal costs)
    #    Souvent, on met un poids plus élevé sur l'état final désiré.
    terminal_costs = CostModelManager(robot_model, robot_data)
    terminal_costs.add_regulation_state_cost(x_ref=x0, weight=100.0, name="terminal_state")

    # 6. Construire le problème
    shooting_problem = problem_builder.build(running_cost_manager=running_costs,
                                             terminal_cost_manager=terminal_costs)

    # 7. Vérification
    print("\n--- Vérification du problème ---")
    print(f"État initial: {shooting_problem.x0.shape}")
    print(f"Nombre de noeuds: {len(shooting_problem.runningModels)}")
    # Vous pouvez maintenant passer `shooting_problem` à un solveur comme DDP.
    # solver = crocoddyl.SolverDDP(shooting_problem)
    # ...
