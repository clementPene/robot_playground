#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from typing import Optional, Callable

# Initialize DDS (Cyclone/FastDDS) factory and use our H1v2 bridge
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config  # expects fields like: ROBOT_SCENE, SIMULATE_DT, DOMAIN_ID, INTERFACE, ...

# ----------------------- Globals (minimaux, comme l’original) -----------------------
locker = threading.Lock()
mj_model: Optional[mujoco.MjModel] = None
mj_data: Optional[mujoco.MjData] = None
viewer = None
elastic_band: Optional[ElasticBand] = None
band_attached_link: int = 0
sim_thread: Optional[Thread] = None
viewer_thread: Optional[Thread] = None
control_cb: Optional[Callable[[mujoco.MjModel, mujoco.MjData, float], Optional[object]]] = None

# ----------------------- Notebook-friendly: construction -----------------------
def make_context(enable_viewer: bool = True, enable_elastic_band: Optional[bool] = None):
    """
    Charge le modèle, crée les buffers de simulation, et (optionnellement) lance le viewer passif.
    À appeler une seule fois avant start().
    """
    global mj_model, mj_data, viewer, elastic_band, band_attached_link

    # Tip (headless servers): uncomment the next line to use EGL without a display
    # os.environ.setdefault("MUJOCO_GL", "egl")

    try:
        mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
    except Exception as e:
        raise SystemExit(f"[FATAL] Failed to load MuJoCo scene '{config.ROBOT_SCENE}': {e}")

    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = float(config.SIMULATE_DT)

    # Optional elastic band
    if enable_elastic_band is None:
        enable_elastic_band = bool(getattr(config, "ENABLE_ELASTIC_BAND", False))
    elastic_band = ElasticBand() if enable_elastic_band else None
    band_attached_link = 0
    if elastic_band is not None:
        try:
            band_attached_link = mj_model.body("torso_link").id
        except Exception:
            print("[WARN] body 'torso_link' not found. Elastic band will attach to body id 0 (world).")

    # Launch passive viewer (optional)
    viewer = None
    if enable_viewer:
        try:
            if elastic_band is not None:
                viewer = mujoco.viewer.launch_passive(
                    mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
                )
            else:
                viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
        except Exception as e:
            raise SystemExit(f"[FATAL] Could not start MuJoCo viewer: {e}")

        # Small pause to let the viewer settle
        time.sleep(0.2)

    # Quick derived sizes (handy to print/debug)
    num_motor_ = mj_model.nu
    dim_motor_sensor_ = 3 * num_motor_  # [q, dq, tau_est] per motor (convention bridge)
    print(f"[INIT] dt={mj_model.opt.timestep:.6f}s | motors(nu)={num_motor_} | expected motor sensor dim={dim_motor_sensor_} | viewer={'on' if viewer else 'off'} | elastic={'on' if elastic_band else 'off'}")

# ----------------------- Threads (inchangés sauf callbacks optionnels) -----------------------
def SimulationThread():
    """
    The physics loop:
      - Initializes the SDK2 channel factory (DDS) ONCE in this thread
      - Starts the H1v2 bridge (SDK2 pub/sub + periodic publishers)
      - Steps the MuJoCo simulation at config.SIMULATE_DT
      - Optionally applies the elastic band force at the attached link
      - Optionally calls a control callback before mj_step (pour le notebook)
      - Sleeps a bit to roughly match real-time
    """
    global mj_data, mj_model, viewer, elastic_band, control_cb

    if mj_model is None or mj_data is None:
        raise SystemExit("[FATAL] make_context() must be called before start().")

    # 1) DDS init (domain/interface come from your config)
    print(f"[INFO] Initializing SDK2 channels (domain={config.DOMAIN_ID}, iface='{config.INTERFACE}')")
    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)

    # 2) Start our H1v2-only bridge (topics handled inside)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)
    print("[INFO] UnitreeSdk2Bridge is up (topics: +rt/lowstate, +rt/wirelesscontroller, -rt/lowcmd)")

    # 3) Optional: joystick support for wireless controller publishing
    if getattr(config, "USE_JOYSTICK", False):
        try:
            device_id = int(getattr(config, "JOYSTICK_DEVICE", 0))
            unitree.SetupJoystick(device_id=device_id, js_type=config.JOYSTICK_TYPE)
            print(f"[INFO] Gamepad initialized as '{config.JOYSTICK_TYPE}' (device {device_id})")
        except SystemExit:
            print("[WARN] No gamepad detected; continuing without it.")
        except Exception as e:
            print(f"[WARN] Gamepad init failed: {e}")

    # 4) Optional: scene printout
    if getattr(config, "PRINT_SCENE_INFORMATION", False):
        unitree.PrintSceneInformation()

    last_log = time.time()
    steps = 0
    start_wall = last_log

    while viewer is None or viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()
        try:
            # Optional: user control callback for notebooks
            if control_cb is not None:
                try:
                    u = control_cb(mj_model, mj_data, mj_data.time)
                    if u is not None:
                        mj_data.ctrl[: mj_model.nu] = u  # pas de clamp, fidèle à ta version
                except Exception as e:
                    # Ne pas casser la boucle si le callback plante
                    print(f"[WARN] control_cb failed: {e}")

            # Optional elastic band
            if elastic_band is not None and elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )

            # Advance physics
            mujoco.mj_step(mj_model, mj_data)
        finally:
            locker.release()

        # Periodic console info (1 Hz)
        now = time.time()
        steps += 1
        if now - last_log >= 1.0:
            sim_t = mj_data.time
            elapsed = now - start_wall
            sps = steps / elapsed if elapsed > 0 else 0.0
            rt_ratio = (sim_t / elapsed) if elapsed > 0 else 0.0
            print(f"[t={sim_t:7.3f}s] steps={steps:6d} | avg {sps:6.1f} steps/s | RT={rt_ratio:4.2f}x")
            last_log = now

        # Sleep to approximately match real time
        time_until_next_step = mj_model.opt.timestep - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    print("[EXIT] Simulation thread finished (viewer closed).")

def PhysicsViewerThread():
    """
    The viewer thread:
      - Calls viewer.sync() regularly so the window stays responsive
      - Uses the same lock so it never reads while the sim thread writes
    """
    global viewer
    if viewer is None:
        return
    view_dt = float(getattr(config, "VIEWER_DT", 0.01))
    while viewer.is_running():
        locker.acquire()
        try:
            viewer.sync()
        finally:
            locker.release()
        time.sleep(view_dt)
    print("[EXIT] Viewer thread finished (window closed).")

# ----------------------- Notebook API -----------------------
def set_control_callback(cb: Optional[Callable[[mujoco.MjModel, mujoco.MjData, float], Optional[object]]]):
    """Enregistre un callback utilisateur: cb(model, data, t) -> (vector nu) ou None."""
    global control_cb
    control_cb = cb

def start():
    """Démarre les threads sim et viewer (si viewer existe)."""
    global sim_thread, viewer_thread, viewer
    if mj_model is None or mj_data is None:
        raise SystemExit("[FATAL] make_context() must be called before start().")
    if sim_thread and sim_thread.is_alive():
        print("[WARN] Simulation already running.")
        return
    viewer_thread = Thread(target=PhysicsViewerThread, name="viewer_thread")
    sim_thread = Thread(target=SimulationThread, name="sim_thread")
    viewer_thread.start()
    sim_thread.start()

def stop():
    """Ferme proprement le viewer et joint les threads."""
    global viewer_thread, sim_thread, viewer
    try:
        if viewer is not None:
            viewer.close()
    except Exception:
        pass
    # Join threads
    for th in (viewer_thread, sim_thread):
        if th and th.is_alive():
            try:
                th.join(timeout=2.0)
            except Exception:
                pass
    viewer_thread = None
    sim_thread = None

def is_running() -> bool:
    """True si la fenêtre viewer est ouverte et/ou la simulation tourne."""
    alive = False
    if sim_thread and sim_thread.is_alive():
        alive = True
    if viewer_thread and viewer_thread.is_alive():
        alive = True
    if viewer is not None and hasattr(viewer, "is_running"):
        alive = alive or viewer.is_running()
    return bool(alive)

def step_once():
    """Un seul pas de physique, sans threads (utile en notebook headless)."""
    if mj_model is None or mj_data is None:
        raise SystemExit("[FATAL] make_context() must be called before step_once().")
    locker.acquire()
    try:
        if control_cb is not None:
            try:
                u = control_cb(mj_model, mj_data, mj_data.time)
                if u is not None:
                    mj_data.ctrl[: mj_model.nu] = u
            except Exception as e:
                print(f"[WARN] control_cb failed: {e}")
        if elastic_band is not None and elastic_band.enable:
            mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                mj_data.qpos[:3], mj_data.qvel[:3]
            )
        mujoco.mj_step(mj_model, mj_data)
        if viewer is not None:
            try:
                viewer.sync()
            except Exception:
                pass
    finally:
        locker.release()

def get_snapshot() -> dict:
    """Copie légère de l’état courant (time, dt, q, v, ctrl)."""
    if mj_model is None or mj_data is None:
        raise SystemExit("[FATAL] make_context() must be called before get_snapshot().")
    locker.acquire()
    try:
        return dict(
            t=float(mj_data.time),
            dt=float(mj_model.opt.timestep),
            q=mj_data.qpos.copy(),
            v=mj_data.qvel.copy(),
            ctrl=mj_data.ctrl.copy(),
        )
    finally:
        locker.release()

# ----------------------- Entrée script (comportement original) -----------------------
if __name__ == "__main__":
    # Charge le modèle et lance le viewer, puis démarre les threads
    make_context(enable_viewer=True, enable_elastic_band=getattr(config, "ENABLE_ELASTIC_BAND", False))
    # Non-daemon + proper join -> clean exit when window closes
    viewer_thread = Thread(target=PhysicsViewerThread, name="viewer_thread")
    sim_thread = Thread(target=SimulationThread, name="sim_thread")
    viewer_thread.start()
    sim_thread.start()
    try:
        viewer_thread.join()
        sim_thread.join()
    except KeyboardInterrupt:
        print("\n[INT] KeyboardInterrupt -> closing viewer…")
        try:
            if viewer is not None:
                viewer.close()
        except Exception:
            pass
