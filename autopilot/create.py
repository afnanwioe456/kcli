from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from time import sleep
from krpc.services.spacecenter import SASMode
from .utils import *

if TYPE_CHECKING:
    from krpc.services.spacecenter import Vessel, Part


def get_closer(distance: float, vessel: Vessel, target: Vessel, rate: float = 0.1):
    sc = UTIL_CONN.space_center
    set_point = distance
    a_limit = -vessel.available_rcs_force[1][1] / vessel.mass * 0.8
    # TODO: rcs_force没有考虑limit设置，只是单纯的最大值
    ref_v_orb = vessel.orbital_reference_frame
    ref_v = vessel.reference_frame
    Kp = np.array([1, 1, 1])
    Ki = np.array([0.05, 0.05, 0.05])
    Kd = np.array([0.1, 0.1, 0.1])
    pid_d = PIDVController(Kp, Ki, Kd)
    prev_time = sc.ut
    stable = 0

    sc.target_vessel = target
    vessel.control.sas = True
    vessel.control.sas_mode = SASMode.target
    while True:
        sleep(rate)
        cur_time = sc.ut
        dt = cur_time - prev_time
        prev_time = cur_time

        position = np.array(target.position(ref_v_orb))
        velocity = -np.array(target.velocity(ref_v_orb))
        d = np.linalg.norm(position)
        dd = d - set_point
        dd_abs = abs(dd)
        s_target = dd / dd_abs * (2 * a_limit * dd_abs) ** 0.5
        s_target = s_target * smooth_step(s_target, 5)  # 10m/s以内快速衰减避免过冲
        v_target = position / d * s_target
        dv = v_target - velocity

        direction = sc.transform_direction(tuple(dv), ref_v_orb, ref_v)
        direction = np.array(direction)
        control = -pid_d.compute(np.zeros(3), direction, dt)
        vessel.control.right = control[0]
        vessel.control.forward = control[1]
        vessel.control.up = -control[2]
        if stable > 30:
            break
        if dd_abs < 5 and abs(s_target) < 0.1:
            stable += 1
        else:
            stable = 0
    vessel.control.sas = False
            