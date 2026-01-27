from typing import Optional

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
    LowState_,
    LowCmd_,
    MotorState_,
    MotorCmd_,
    IMUState_,
)


def _fmt_array(arr, precision=4):
    """Format a numeric sequence into a compact string."""
    try:
        return "[" + ", ".join(f"{float(x): .{precision}f}" for x in arr) + "]"
    except Exception:
        return str(arr)


def print_low_state(ls: Optional[LowState_], max_motors: int = 35) -> None:
    """
    Pretty-print the raw content of a LowState_ message.

    - ls : LowState_ instance
    - max_motors : maximum number of motor_state entries to display
    """
    if ls is None:
        print("LowState_ = None")
        return

    print("=== LowState_ (raw DDS) ===")

    # Header
    try:
        ver = getattr(ls, "version", None)
        if ver is not None and len(ver) >= 2:
            print(f"version       : [{ver[0]}, {ver[1]}]")
    except Exception:
        pass

    print(f"mode_pr       : {ls.mode_pr}")
    print(f"mode_machine  : {ls.mode_machine}")

    if hasattr(ls, "tick"):
        print(f"tick          : {ls.tick}")

    # IMU
    imu: IMUState_ = ls.imu_state
    print("--- IMUState_ ---")
    print(f"  quaternion   : {_fmt_array(imu.quaternion, precision=4)}")
    print(f"  gyroscope    : {_fmt_array(imu.gyroscope,   precision=4)}")
    print(f"  accelerom.   : {_fmt_array(imu.accelerometer, precision=4)}")
    print(f"  rpy          : {_fmt_array(imu.rpy, precision=4)}")
    print(f"  temperature  : {imu.temperature}")

    # Motors
    n_mot = len(ls.motor_state)
    print(f"\n--- MotorState_ ---")
    print(f"num motors     : {n_mot}")

    n = min(n_mot, max_motors)
    print("idx | mode |        q |       dq |      ddq |  tau_est |  temp[0]  temp[1] |      vol | sensor[0]  sensor[1] | motorstate")
    print("-" * 120)
    for i in range(n):
        ms: MotorState_ = ls.motor_state[i]
        t0 = ms.temperature[0] if len(ms.temperature) > 0 else None
        t1 = ms.temperature[1] if len(ms.temperature) > 1 else None
        s0 = ms.sensor[0] if len(ms.sensor) > 0 else None
        s1 = ms.sensor[1] if len(ms.sensor) > 1 else None

        print(
            f"{i:3d} | {ms.mode:4d} |"
            f" {ms.q:8.4f} | {ms.dq:8.4f} | {ms.ddq:8.4f} |"
            f" {ms.tau_est:8.4f} |"
            f" {t0!s:7}  {t1!s:7} |"
            f" {ms.vol:8.4f} |"
            f" {s0!s:8}  {s1!s:8} |"
            f" {ms.motorstate}"
        )

    if n_mot > n:
        print(f"... ({n_mot - n} additional motors not shown)")

    # Wireless remote (just size and a short preview)
    if hasattr(ls, "wireless_remote"):
        wr = ls.wireless_remote
        print(f"\n--- wireless_remote ---")
        print(f"size          : {len(wr)}")
        if len(wr) > 0:
            preview = " ".join(f"{int(b):02x}" for b in wr[:16])
            print(f"first bytes (hex) : {preview} ...")

    # Reserve + CRC
    if hasattr(ls, "reserve"):
        print(f"\nreserve       : {list(ls.reserve)}")
    if hasattr(ls, "crc"):
        print(f"crc           : {ls.crc}")

    print("============================\n")


def print_low_cmd(cmd: Optional[LowCmd_], max_motors: int = 35) -> None:
    """
    Pretty-print the raw content of a LowCmd_ message.

    - cmd : LowCmd_ instance
    - max_motors : maximum number of motor_cmd entries to display
    """
    if cmd is None:
        print("LowCmd_ = None")
        return

    print("=== LowCmd_ (raw DDS) ===")

    print(f"mode_pr       : {cmd.mode_pr}")
    print(f"mode_machine  : {cmd.mode_machine}")

    # Motors
    n_mot = len(cmd.motor_cmd)
    print(f"\n--- MotorCmd_ ---")
    print(f"num motors     : {n_mot}")

    n = min(n_mot, max_motors)
    print("idx | mode |        q |       dq |      tau |       kp |       kd | reserve")
    print("-" * 90)
    for i in range(n):
        mc: MotorCmd_ = cmd.motor_cmd[i]
        print(
            f"{i:3d} | {mc.mode:4d} |"
            f" {mc.q:8.4f} | {mc.dq:8.4f} | {mc.tau:8.4f} |"
            f" {mc.kp:8.4f} | {mc.kd:8.4f} | {mc.reserve}"
        )

    if n_mot > n:
        print(f"... ({n_mot - n} additional motors not shown)")

    # Reserve + CRC
    if hasattr(cmd, "reserve"):
        print(f"\nreserve       : {list(cmd.reserve)}")
    if hasattr(cmd, "crc"):
        print(f"crc           : {cmd.crc}")

    print("============================\n")
