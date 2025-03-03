from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.optimize import linprog
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
    step = 0

    sc.target_vessel = target
    vessel.control.sas = True
    sleep(0.1)
    vessel.control.sas_mode = SASMode.target
    # TODO: 稳定判断
    sleep(10)
    balance_rcs(vessel)
    while True:
        sleep(rate)
        step += 1
        cur_time = sc.ut
        dt = cur_time - prev_time
        prev_time = cur_time

        position = np.array(target.position(ref_v_orb))
        velocity = -np.array(target.velocity(ref_v_orb))
        d = np.linalg.norm(position)
        dd = d - set_point
        dd_abs = abs(dd)
        s_target = dd / dd_abs * (2 * a_limit * dd_abs) ** 0.5
        s_target = s_target * smooth_step(s_target, 10)  # 20m/s以内快速衰减避免过冲
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
        if step % 100 == 0:
            balance_rcs(vessel)
    vessel.control.sas = False

            
def balance_rcs(v: Vessel):
    force = []
    torque = []
    for p in v.parts.rcs:
        p_force = p.available_force
        p_torque = p.available_torque
        force.append(p_force[0] + tuple(-x for x in p_force[1]))
        torque.append(tuple(x1 + x2 for x1, x2 in zip(p_torque[0], p_torque[1])))
    force = np.where(np.abs(force) < 0.1, 0, force)
    torque = np.where(np.abs(torque) < 0.01, 0, torque)
    force = np.maximum(np.array(force).T, 0.)
    torque = np.array(torque).T
    eq = np.zeros(torque.shape[0])
    c = -np.ones(force.shape[1])
    control = np.zeros(force.shape[1])
    for d in range(force.shape[0]):
        # for each (6) thrust direction
        bounds = [(0, max_t) for max_t in force[d]]
        result = linprog(c, A_eq=torque, b_eq=eq, bounds=bounds, method='highs')
        limit = np.array([t for _, t in enumerate(result.x)])
        if max(limit) < 0.1:
            return
        limit /= max(limit)
        control = np.maximum(control, limit)
    for p, c in zip(v.parts.rcs, control):
        p.thrust_limit = c
