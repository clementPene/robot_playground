# ===============================================================
#  MuJoCo <-> Unitree SDK2 bridge for H1v2 (Humanoid v2)
#  - Publishes ONLY:
#       * rt/lowstate           (low-level state: joints + IMU)
#       * rt/wirelesscontroller (gamepad state)
#  - Subscribes ONLY:
#       * rt/lowcmd             (low-level motor command)
#  - HighState was removed on purpose.
# ---------------------------------------------------------------
#  Super short mental model:
#   - MuJoCo = physics simulator. You push torques in, it moves
#     the robot and gives you positions/velocities/forces out.
#   - Unitree SDK2 = message bus (publish/subscribe) so components
#     can talk. Here we:
#       1) receive LowCmd (desired torques/PD setpoints),
#       2) apply them to MuJoCo,
#       3) read joint state (+ IMU),
#       4) publish LowState,
#       5) (optional) publish gamepad state for teleop.
# ===============================================================

import mujoco
import numpy as np
import pygame
import sys
import struct

# --- SDK2: channels + message types ---
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher

# Gamepad message lives under unitree_go
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_

# H1v2 uses the "hg" (humanoid generation) IDL for low-level cmd/state
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default

# Run a function at fixed rate (simple periodic thread) provided by SDK2
from unitree_sdk2py.utils.thread import RecurrentThread

# ------------------ TOPIC NAMES (we keep only these 3) -----------------
TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_WIRELESS_CONTROLLER = "rt/wirelesscontroller"

# ------------------ SIMPLE CONSTANTS -----------------------------------
# We assume each motor exposes 3 numbers in sensordata: q, dq, tau_est
MOTOR_SENSOR_NUM = 3

# For H1/H2 "hg" IDL, motor arrays are large (e.g., 35). We’ll fill the
# first 'num_motor' entries; the message can be larger, that’s fine.
NUM_MOTOR_IDL_HG = 35


class UnitreeSdk2Bridge:
    """
    Glue between MuJoCo and SDK2:
      - Subscribe LowCmd -> turn that into actuator commands in MuJoCo
      - Publish LowState -> current joint state (+ IMU) from MuJoCo
      - Publish WirelessController -> current gamepad state
    """

    def __init__(self, mj_model, mj_data):
        # Pointers to MuJoCo:
        #   - mj_model = static info (sizes, names, options)
        #   - mj_data  = live state (qpos, qvel, sensordata,...)
        self.mj_model = mj_model
        self.mj_data = mj_data

        # How many actuators (commanded motors) are in the MuJoCo model
        self.num_motor = self.mj_model.nu

        # Size of the "motor sensor block" we expect at the start of sensordata
        # (q, dq, tau_est) per motor
        self.dim_motor_sensor = MOTOR_SENSOR_NUM * self.num_motor

        # Flags telling us what sensors exist in the MuJoCo scene
        self.have_imu = False          # IMU (quat + gyro + acc)
        self.have_frame_sensor = False # example: "frame_pos" (unused here)

        # Simulation time step (e.g., 0.001 s = 1 kHz)
        self.dt = self.mj_model.opt.timestep

        # Gamepad handle (pygame)
        self.joystick = None

        # =============== Discover what sensors are present ===============
        # We loop over declared sensors and check their names from the XML.
        for i in range(self.mj_model.nsensor):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i)
            if name == "imu_quat":
                self.have_imu = True
            if name == "frame_pos":
                self.have_frame_sensor = True

        # =================== Set up publishers/subscribers ===================

        # 1) LowState publisher -> "rt/lowstate"
        #    LowState_default() gives us a correctly-shaped message for H1.
        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self.low_state_puber.Init()

        # Send LowState periodically at the MuJoCo simulation rate
        self.lowStateThread = RecurrentThread(
            interval=self.dt, target=self.PublishLowState, name="sim_lowstate"
        )
        self.lowStateThread.Start()

        # 2) WirelessController publisher -> "rt/wirelesscontroller"
        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher(
            TOPIC_WIRELESS_CONTROLLER, WirelessController_
        )
        self.wireless_controller_puber.Init()

        # Publish gamepad at 100 Hz
        self.WirelessControllerThread = RecurrentThread(
            interval=0.01,
            target=self.PublishWirelessController,
            name="sim_wireless_controller",
        )
        self.WirelessControllerThread.Start()

        # 3) LowCmd subscriber -> "rt/lowcmd"
        #    Every time we receive a command, LowCmdHandler is called.
        self.low_cmd_suber = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 10)  # queue size = 10

        # =============== Gamepad button map (to Unitree bitfield) ===============
        # WirelessController packs 16 buttons in an int: bit i = button i
        self.key_map = {
            "R1": 0, "L1": 1, "start": 2, "select": 3,
            "R2": 4, "L2": 5, "F1": 6, "F2": 7,
            "A": 8, "B": 9, "X": 10, "Y": 11,
            "up": 12, "right": 13, "down": 14, "left": 15,
        }

    # --------------------------------------------------------------------------
    # 1) Receive motor commands (LowCmd) and apply them in MuJoCo
    # --------------------------------------------------------------------------
    def LowCmdHandler(self, msg: LowCmd_):
        """
        LowCmd holds, for EACH motor:
          - tau  : feedforward torque
          - kp,kd: PD gains
          - q,dq : desired position and velocity (rad, rad/s)
        We do a simple PD+tau:
            ctrl[i] = tau + kp*(q_des - q_meas) + kd*(dq_des - dq_meas)
        Where q_meas and dq_meas are read from sensordata.
        ctrl[i] is what we send to actuator i in MuJoCo.
        """
        if self.mj_data is not None:
            for i in range(self.num_motor):
                # We assume sensordata layout: [q(0..n-1), dq(0..n-1), tau_est(0..n-1), ...]
                q_meas = self.mj_data.sensordata[i]
                dq_meas = self.mj_data.sensordata[i + self.num_motor]
                cmd = msg.motor_cmd[i]
                self.mj_data.ctrl[i] = (
                    cmd.tau
                    + cmd.kp * (cmd.q - q_meas)
                    + cmd.kd * (cmd.dq - dq_meas)
                )

    # --------------------------------------------------------------------------
    # 2) Publish low-level state (LowState)
    # --------------------------------------------------------------------------
    def PublishLowState(self):
        """
        Read MuJoCo state and fill LowState:
          - motor_state[i].q       = joint position (rad)
          - motor_state[i].dq      = joint velocity (rad/s)
          - motor_state[i].tau_est = estimated torque (N.m) from sim
        + IMU if present:
          - quaternion[4], gyroscope[3], accelerometer[3]
        Then publish the message.
        """
        if self.mj_data is not None:
            # --- MOTORS: assume joint sensors are packed first in sensordata
            for i in range(self.num_motor):
                self.low_state.motor_state[i].q = self.mj_data.sensordata[i]
                self.low_state.motor_state[i].dq = self.mj_data.sensordata[i + self.num_motor]
                self.low_state.motor_state[i].tau_est = self.mj_data.sensordata[i + 2 * self.num_motor]

            # --- IMU: write only if "imu_quat" sensor exists
            # Simplified assumption: IMU block comes right after motors:
            # quat(4), gyro(3), acc(3) = 10 values total.
            if self.have_imu:
                base = self.dim_motor_sensor
                if len(self.mj_data.sensordata) >= base + 10:
                    self.low_state.imu_state.quaternion[0] = self.mj_data.sensordata[base + 0]
                    self.low_state.imu_state.quaternion[1] = self.mj_data.sensordata[base + 1]
                    self.low_state.imu_state.quaternion[2] = self.mj_data.sensordata[base + 2]
                    self.low_state.imu_state.quaternion[3] = self.mj_data.sensordata[base + 3]

                    self.low_state.imu_state.gyroscope[0] = self.mj_data.sensordata[base + 4]
                    self.low_state.imu_state.gyroscope[1] = self.mj_data.sensordata[base + 5]
                    self.low_state.imu_state.gyroscope[2] = self.mj_data.sensordata[base + 6]

                    self.low_state.imu_state.accelerometer[0] = self.mj_data.sensordata[base + 7]
                    self.low_state.imu_state.accelerometer[1] = self.mj_data.sensordata[base + 8]
                    self.low_state.imu_state.accelerometer[2] = self.mj_data.sensordata[base + 9]

            # Done: push to the topic
            self.low_state_puber.Write(self.low_state)

    # --------------------------------------------------------------------------
    # 3) Publish gamepad state (WirelessController)
    # --------------------------------------------------------------------------
    def PublishWirelessController(self):
        """
        If a gamepad is connected, read its buttons/axes via pygame
        and publish WirelessController (Unitree format).
        """
        if self.joystick is not None:
            # If user forgot to call SetupJoystick, don't crash
            if not hasattr(self, "axis_id") or not hasattr(self, "button_id"):
                return

            pygame.event.get()

            # 16-button bitfield (1 = pressed, 0 = not pressed)
            key_state = [0] * 16
            key_state[self.key_map["R1"]] = self.joystick.get_button(self.button_id["RB"])
            key_state[self.key_map["L1"]] = self.joystick.get_button(self.button_id["LB"])
            key_state[self.key_map["start"]] = self.joystick.get_button(self.button_id["START"])
            key_state[self.key_map["select"]] = self.joystick.get_button(self.button_id["SELECT"])
            key_state[self.key_map["R2"]] = int(self.joystick.get_axis(self.axis_id["RT"]) > 0)
            key_state[self.key_map["L2"]] = int(self.joystick.get_axis(self.axis_id["LT"]) > 0)
            key_state[self.key_map["F1"]] = 0
            key_state[self.key_map["F2"]] = 0
            key_state[self.key_map["A"]] = self.joystick.get_button(self.button_id["A"])
            key_state[self.key_map["B"]] = self.joystick.get_button(self.button_id["B"])
            key_state[self.key_map["X"]] = self.joystick.get_button(self.button_id["X"])
            key_state[self.key_map["Y"]] = self.joystick.get_button(self.button_id["Y"])
            key_state[self.key_map["up"]] = int(self.joystick.get_hat(0)[1] > 0)
            key_state[self.key_map["right"]] = int(self.joystick.get_hat(0)[0] > 0)
            key_state[self.key_map["down"]] = int(self.joystick.get_hat(0)[1] < 0)
            key_state[self.key_map["left"]] = int(self.joystick.get_hat(0)[0] < 0)

            # Pack bits into an integer
            key_value = 0
            for i in range(16):
                key_value |= (key_state[i] << i)

            # Sticks (left/right). Many UIs prefer +Y up; pygame gives +Y down, so we negate LY/RY.
            self.wireless_controller.keys = key_value
            self.wireless_controller.lx = self.joystick.get_axis(self.axis_id["LX"])
            self.wireless_controller.ly = -self.joystick.get_axis(self.axis_id["LY"])
            self.wireless_controller.rx = self.joystick.get_axis(self.axis_id["RX"])
            self.wireless_controller.ry = -self.joystick.get_axis(self.axis_id["RY"])

            # Publish
            self.wireless_controller_puber.Write(self.wireless_controller)

    # --------------------------------------------------------------------------
    # 4) Gamepad setup (optional)
    # --------------------------------------------------------------------------
    def SetupJoystick(self, device_id=0, js_type="xbox"):
        """
        If you have no gamepad: ignore this.
        Otherwise:
          - plug the gamepad,
          - js_type = "xbox" or "switch" to match your pad,
          - device_id = 0 if you have only one.
        """
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(device_id)
            self.joystick.init()
        else:
            print("No gamepad detected.")
            sys.exit()

        if js_type == "xbox":
            self.axis_id = {  # axis indices in pygame for Xbox-style pads
                "LX": 0, "LY": 1, "RX": 3, "RY": 4, "LT": 2, "RT": 5, "DX": 6, "DY": 7,
            }
            self.button_id = {  # button indices in pygame for Xbox-style pads
                "X": 2, "Y": 3, "B": 1, "A": 0, "LB": 4, "RB": 5, "SELECT": 6, "START": 7,
            }
        elif js_type == "switch":
            self.axis_id = {
                "LX": 0, "LY": 1, "RX": 2, "RY": 3, "LT": 5, "RT": 4, "DX": 6, "DY": 7,
            }
            self.button_id = {
                "X": 3, "Y": 4, "B": 1, "A": 0, "LB": 6, "RB": 7, "SELECT": 10, "START": 11,
            }
        else:
            print("Unsupported gamepad.")

    # --------------------------------------------------------------------------
    # 5) Utility: print MuJoCo scene info (handy for debugging)
    # --------------------------------------------------------------------------
    def PrintSceneInformation(self):
        """
        Lists names and indices of bodies, joints, actuators, and sensors.
        Super useful to learn "who is who" in your MJCF.
        """
        print(" ")

        print("<<------------- Link (body) ------------->> ")
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, i)
            if name:
                print("link_index:", i, ", name:", name)
        print(" ")

        print("<<------------- Joint ------------->> ")
        for i in range(self.mj_model.njnt):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_JOINT, i)
            if name:
                print("joint_index:", i, ", name:", name)
        print(" ")

        print("<<------------- Actuator ------------->>")
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                print("actuator_index:", i, ", name:", name)
        print(" ")

        print("<<------------- Sensor ------------->>")
        index = 0
        for i in range(self.mj_model.nsensor):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i)
            if name:
                print(
                    "sensor_index:",
                    index,
                    ", name:",
                    name,
                    ", dim:",
                    self.mj_model.sensor_dim[i],
                )
            index = index + self.mj_model.sensor_dim[i]
        print(" ")


# --------------------------------------------------------------------------
# Optional: a simple "rubber band" helper to pull the robot (testing tool)
# --------------------------------------------------------------------------
class ElasticBand:
    def __init__(self):
        self.stiffness = 200
        self.damping = 100
        self.point = np.array([0, 0, 3])
        self.length = 0
        self.enable = True

    def Advance(self, x, dx):
        """
        x  = current 3D position of some robot point
        dx = current 3D velocity of that point
        Returns a spring-damper force pulling towards self.point.
        """
        δx = self.point - x
        distance = np.linalg.norm(δx)
        if distance < 1e-8:
            return np.zeros(3)
        direction = δx / distance
        v = np.dot(dx, direction)
        f = (self.stiffness * (distance - self.length) - self.damping * v) * direction
        return f

    def MujuocoKeyCallback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable
