from robot_deploy.controllers.policy import Policy




class MPCPolicy(Policy):
    def __init__(...):
        # tout ce que tu fais aujourd’hui AVANT la boucle for i in range(N_OCP_STEPS)
        # doit aller ici (construction du modèle, OCP, solver, warm start, etc.)
        ...

    def step(self, state: dict, command: np.ndarray):
        # 1) convertir l’état robot_deploy -> x_current (q, v)
        # 2) mettre à jour x0 et les références du problème OCP
        # 3) résoudre le MPC (quelques itérations)
        # 4) prendre u_optimal = solver.us[0]
        # 5) retourner (dt, q_ref, dq_ref, kps, kds)
        ...

    def save_data(self, log_dir=None):
        # si tu veux logguer ocp_save, physics_log, etc.
        ...