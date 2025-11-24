import time
import argparse
import yaml
import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

import numpy as np

JOINT_INDEX_TO_NAME = [
    "left_hip_yaw_joint","left_hip_pitch_joint","left_hip_roll_joint","left_knee_joint",
    "left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_yaw_joint","right_hip_pitch_joint","right_hip_roll_joint","right_knee_joint",
    "right_ankle_pitch_joint","right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
]
NAME_TO_INDEX = {n: i for i, n in enumerate(JOINT_INDEX_TO_NAME)}
H1_2_NUM_MOTOR = len(JOINT_INDEX_TO_NAME)
    

def load_cfg(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    # contrôleur
    ctrl = cfg.get("controller", {})
    ctrl.setdefault("control_dt", 0.002)
    ctrl.setdefault("mode_pr", "PR")  # "PR" ou "AB"
    ctrl.setdefault("mode_machine", "inherit")  # "inherit" ou int
    ctrl.setdefault("ramp_to_zero_duration", 2.0)  # s
    cfg["controller"] = ctrl
    # joints
    cfg.setdefault("joints", [])
    # map rapide name -> dict
    joints_map = {j["name"]: j for j in cfg["joints"]}
    cfg["_joints_map"] = joints_map
    return cfg

class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints

class Custom:
    def __init__(self, cfg):
        self.cfg = cfg
        ctrl = cfg["controller"]

        self.time_ = 0.0
        self.control_dt_ = float(ctrl["control_dt"])
        self.ramp_duration_ = float(ctrl.get("ramp_to_zero_duration", 2.0))


        self.mode_pr_ = Mode.PR if str(ctrl["mode_pr"]).upper() == "PR" else Mode.AB
        self.mode_machine_cfg_ = ctrl["mode_machine"]
        self.mode_machine_ = 0

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.crc = CRC()

        # posture initiale lue au 1er LowState pour faire un ramp vers 0
        self.q_init_ = None
        self.got_first_state_ = False

        # table des gains/flags
        self.joint_cfg_map_ = cfg["_joints_map"]

        # compteur print
        self.counter_ = 0

    def Init(self):
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

        # initialisation des états la première fois
        if not self.got_first_state_:
            # posture initiale (point de départ du ramp)
            self.q_init_ = [msg.motor_state[i].q for i in range(H1_2_NUM_MOTOR)]
            # mode_machine: hérite ou fixe
            if isinstance(self.mode_machine_cfg_, str) and self.mode_machine_cfg_.lower() == "inherit":
                self.mode_machine_ = int(msg.mode_machine)
            else:
                self.mode_machine_ = int(self.mode_machine_cfg_)
            self.got_first_state_ = True

        # affichage IMU 1 Hz
        self.counter_ += 1
        if (self.counter_ % 500 == 0) :
            self.counter_ = 0
            print(self.low_state.imu_state.rpy)

    def Start(self):
        # attendre le premier LowState
        while not self.got_first_state_:
            time.sleep(0.001)

        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.LowCmdWrite, name="control"
        )
        self.lowCmdWriteThreadPtr.Start()

    def LowCmdWrite(self):
        # en-têtes
        self.low_cmd.mode_pr = self.mode_pr_
        self.low_cmd.mode_machine = self.mode_machine_

        # ratio de rampe vers 0 (cubic pour éviter à-coups)
        t = min(max(self.time_ / self.ramp_duration_, 0.0), 1.0)
        # spline cubique (0->0, 1->1, dérivées nulles aux bornes)
        ratio = 3*t**2 - 2*t**3 if self.ramp_duration_ > 1e-6 else 1.0

        for i in range(H1_2_NUM_MOTOR):
            name = JOINT_INDEX_TO_NAME[i]
            jcfg = self.joint_cfg_map_.get(name, {})
            enabled = jcfg.get("enabled", True)

            mc = self.low_cmd.motor_cmd[i]

            if not enabled:
                # désactiver le joint
                mc.mode = 0
                mc.tau = 0.0
                mc.kp = 0.0
                mc.kd = 0.0
                continue

            # activer + gains
            mc.mode = 1  # enable
            mc.kp = float(jcfg.get("kp", 0.0))
            mc.kd = float(jcfg.get("kd", 0.0))

            # consigne vers zéro absolu
            q0 = self.q_init_[i] if self.q_init_ is not None else 0.0
            q_ref = (1.0 - ratio) * q0 + ratio * 0.0

            mc.q = q_ref
            mc.dq = 0.0
            mc.tau = 0.0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

        self.time_ += self.control_dt_

if __name__ == '__main__':
    ChannelFactoryInitialize(79, "lo")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    cfg = load_cfg(args.config)

    print("Holding zero position")

    custom = Custom(cfg)
    custom.Init()
    custom.Start()

    while True:        
        time.sleep(1)