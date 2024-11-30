from __future__ import annotations
import numpy as np
import sympy as sp
from astropy import units as u
from typing import TYPE_CHECKING

from ..utils import sec_to_date

if TYPE_CHECKING:
    from .create import Orbit

def angle_between_vectors(vector_1, vector_2):
    return np.arccos(np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2)))


def orbit_launch_window(orbit: Orbit, site_position: np.ndarray, direction='SE', phase_diff=40, start_period=0, end_period=30):
    """
    发射到target轨道面的发射窗口
    """
    period = orbit._poliorbit.attractor.rotational_period.to_value(u.s)
    # 获得目标轨道面的法向量
    target_position = orbit.r
    target_velocity = orbit.v
    target_raan = orbit.raan
    orbit_plane_normal = np.cross(target_position, target_velocity)
    # TODO: 轨道倾角小于发射场纬度时
    # 计算重合时的位置向量(x1, y1, z) (x2, y2, z)
    a, b, c = orbit_plane_normal
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    z = site_position[2]
    result_value = sp.solve([a * x + b * y + c * z,
                             x ** 2 + y ** 2 + z ** 2 - np.linalg.norm(site_position) ** 2],
                            [x, y])
    # 向赤道面投影并计算向量夹角，由周期确定重合倒计时
    result_launch_window = []
    an_p = np.array([np.cos(target_raan), np.sin(target_raan), 0])  # 升交点赤经方向
    for i in list(range(len(result_value))):
        proj_result_p = np.array([float(result_value[i][0]), float(result_value[i][1]), 0])
        proj_site_p = np.array([site_position[0], site_position[1], 0])
        # 排除方向错误的解
        site_arg = angle_between_vectors(an_p, proj_result_p)
        if site_arg > np.pi / 2 and direction == 'NE':
            continue
        if site_arg <= np.pi / 2 and direction == 'SE':
            continue
        phase_difference = angle_between_vectors(proj_result_p, proj_site_p)
        if np.cross(proj_site_p, proj_result_p)[2] < 0:
            phase_difference = 2 * np.pi - phase_difference
        time_difference = phase_difference * period / (2 * np.pi)
        result_launch_window.append(time_difference)

    # 计算最佳发射窗口
    best_phase_angle = 180
    best_launch_window = result_launch_window[0]
    for i in list(range(len(result_launch_window))):
        launch_window = result_launch_window[i] + start_period * period
        launch_window_position = np.array([float(result_value[i][0]), float(result_value[i][1]), z])
        for _ in range(start_period, end_period):
            target_position = orbit.propagate(launch_window).r
            phase_angle = np.degrees(angle_between_vectors(launch_window_position, target_position))
            direction = np.cross(target_position, target_position - launch_window_position) / orbit_plane_normal > 0
            if not direction.all():
                phase_angle = -phase_angle
            if phase_diff < phase_angle < best_phase_angle:
                best_phase_angle = phase_angle
                best_launch_window = launch_window
            launch_window += period

    return best_launch_window + orbit.epoch_sec