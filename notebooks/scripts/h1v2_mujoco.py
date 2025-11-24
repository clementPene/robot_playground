# ===============================================================
#  H1v2 test launcher for MuJoCo + Unitree SDK2
#  - Loads your MuJoCo scene (config.ROBOT_SCENE)
#  - Starts the SDK2 bridge (H1v2-only) from unitree_sdk2py_bridge.py
#  - Opens the interactive viewer (passive) and runs physics in a thread
#  - Optional "ElasticBand" helper to tug the robot for quick testing
#  - Prints useful info in the terminal while running
#
#  Big picture:
#   - This file is the runner. It creates:
#       1) a SimulationThread -> steps physics at dt = config.SIMULATE_DT
#       2) a PhysicsViewerThread -> keeps the viewer responsive (v.sync())
#   - The UnitreeSdk2Bridge (imported from unitree_sdk2py_bridge.py) handles
#     all SDK2 topics: subscribe rt/lowcmd, publish rt/lowstate + wirelesscontroller.
#
#  Assumptions:
#   - H1v2 robot (humanoid). We attach the optional elastic band to "torso_link".
#   - No HighState topic here. Only the 3 topics used by the bridge.
# ===============================================================

import time
import os
import mujoco
import mujoco.viewer
from threading import Thread
import threading

# Initialize DDS (Cyclone/FastDDS) factory and use our H1v2 bridge
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config  # expects fields like: ROBOT_SCENE, SIMULATE_DT, DOMAIN_ID, INTERFACE, ...

# ------------- Global lock to avoid stepping while the viewer reads data -------------
locker = threading.Lock()

# ------------- Load MuJoCo model and data -------------
# Tip (headless servers): uncomment the next line to use EGL without a display
# os.environ.setdefault("MUJOCO_GL", "egl")

try:
    mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
except Exception as e:
    raise SystemExit(f"[FATAL] Failed to load MuJoCo scene '{config.ROBOT_SCENE}': {e}")

mj_data = mujoco.MjData(mj_model)

# Set simulation timestep BEFORE launching the viewer, so everything is consistent
mj_model.opt.timestep = float(config.SIMULATE_DT)

# ------------- Optional: elastic band helper -------------
# For H1v2 we try to attach to "torso_link". If not found, we fallback to base (id=0) with a warning.
elastic_band = None
band_attached_link = 0  # world/body 0 fallback
if getattr(config, "ENABLE_ELASTIC_BAND", False):
    elastic_band = ElasticBand()
    try:
        band_attached_link = mj_model.body("torso_link").id  # H1v2 default link name
    except Exception:
        print("[WARN] body 'torso_link' not found. Elastic band will attach to body id 0 (world).")

# ------------- Launch the passive viewer -------------
# The passive viewer lets us drive the simulation from our own thread.
try:
    if elastic_band is not None:
        viewer = mujoco.viewer.launch_passive(
            mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
        )
    else:
        viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
except Exception as e:
    raise SystemExit(f"[FATAL] Could not start MuJoCo viewer: {e}")

# ------------- Quick derived sizes (handy to print/debug) -------------
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_  # we expect [q, dq, tau_est] per motor

# Small pause to let the viewer settle
time.sleep(0.2)

def SimulationThread():
    """
    The physics loop:
      - Initializes the SDK2 channel factory (DDS) ONCE in this thread
      - Starts the H1v2 bridge (SDK2 pub/sub + periodic publishers)
      - Steps the MuJoCo simulation at config.SIMULATE_DT
      - Optionally applies the elastic band force at the attached link
      - Sleeps a bit to roughly match real-time
    """
    global mj_data, mj_model

    # 1) DDS init (domain/interface come from your config)
    print(f"[INFO] Initializing SDK2 channels (domain={config.DOMAIN_ID}, iface='{config.INTERFACE}')")
    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)

    # 2) Start our H1v2-only bridge (topics handled inside)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)
    print("[INFO] UnitreeSdk2Bridge is up (topics: +rt/lowstate, +rt/wirelesscontroller, -rt/lowcmd)")

    # 3) Optional: joystick support for wireless controller publishing
    if getattr(config, "USE_JOYSTICK", False):
        try:
            unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
            print(f"[INFO] Gamepad initialized as '{config.JOYSTICK_TYPE}'")
        except SystemExit:
            print("[WARN] No gamepad detected; continuing without it.")
        except Exception as e:
            print(f"[WARN] Gamepad init failed: {e}")

    # 4) Optional: scene printout (names, indices, addresses) — very useful the first time
    if getattr(config, "PRINT_SCENE_INFORMATION", False):
        unitree.PrintSceneInformation()

    # 5) Physics step loop
    last_log = time.time()
    steps = 0
    start_wall = last_log

    print(f"[RUN] dt={mj_model.opt.timestep:.6f}s | motors(nu)={num_motor_} | expected motor sensor dim={dim_motor_sensor_}")

    while viewer.is_running():
        step_start = time.perf_counter()

        # ---- lock while we modify/read mj_data so the viewer doesn't read half-updated values
        locker.acquire()
        try:
            # Apply elastic band as a world-frame force on the chosen body (first 3 components)
            if elastic_band is not None and elastic_band.enable:
                # For a quick test we use the root's position/velocity components (qpos[:3], qvel[:3])
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )

            # Advance physics by one step
            mujoco.mj_step(mj_model, mj_data)
        finally:
            locker.release()

        # ---- periodic console info (1 Hz)
        now = time.time()
        steps += 1
        if now - last_log >= 1.0:
            sim_t = mj_data.time
            elapsed = now - start_wall
            sps = steps / elapsed if elapsed > 0 else 0.0  # steps per second
            rt_ratio = (sim_t / elapsed) if elapsed > 0 else 0.0
            print(f"[t={sim_t:7.3f}s] steps={steps:6d} | avg {sps:6.1f} steps/s | RT={rt_ratio:4.2f}x")
            last_log = now

        # ---- sleep to approximately match real time (optional but nice for stability)
        time_until_next_step = mj_model.opt.timestep - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    # When the viewer window is closed, we exit the loop and the thread ends.
    print("[EXIT] Simulation thread finished (viewer closed).")


def PhysicsViewerThread():
    """
    The viewer thread:
      - Calls viewer.sync() regularly so the window stays responsive
      - Uses the same lock so it never reads while the sim thread writes
    """
    view_dt = getattr(config, "VIEWER_DT", 0.01)  # default ~100 Hz
    while viewer.is_running():
        locker.acquire()
        try:
            viewer.sync()
        finally:
            locker.release()
        time.sleep(view_dt)
    print("[EXIT] Viewer thread finished (window closed).")


# ------------- Entry point: start and join threads -------------
if __name__ == "__main__":
    # Create and start threads
    viewer_thread = Thread(target=PhysicsViewerThread, name="viewer_thread")
    sim_thread = Thread(target=SimulationThread, name="sim_thread")

    # Non-daemon + proper join -> clean exit when window closes
    viewer_thread.start()
    sim_thread.start()

    try:
        viewer_thread.join()
        sim_thread.join()
    except KeyboardInterrupt:
        print("\n[INT] KeyboardInterrupt -> closing viewer…")
        try:
            viewer.close()
        except Exception:
            pass
