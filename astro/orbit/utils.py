from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

from ...math import (
    rotation as mr,
    vector as mv
)
from ..utils import UTIL_CONN

if TYPE_CHECKING:
    from .create import Orbit


def orbit_launch_window(orbit: Orbit, 
                        site_coord: tuple[float, float], 
                        direction: str, 
                        cloest: bool,
                        search: bool,
                        min_phase: float, 
                        max_phase: float,
                        start_time: float, 
                        end_time: float):
    """
    发射到target轨道面的发射窗口
    """
    body            = orbit.attractor
    period          = body.rotational_period
    # bci右手系
    # 获得目标轨道面的法向量
    target_position = orbit.r_vec
    target_raan     = orbit.raan
    an_p            = np.array([np.cos(target_raan), np.sin(target_raan), 0], dtype=np.float64)  # 升交点方向
    orbit_plane_n   = orbit.h_vec
    body_n          = body.angular_velocity
    body_n          = mv.normalize(body_n)  # 数值稳定
    # 计算重合时的旋转角
    site_position   = body.surface_position(site_coord, start_time)
    result_rotation = mr.solve_rotation_angle(site_position, orbit_plane_n, body_n, np.pi / 2)
    if result_rotation is None:
        raise ValueError('无解, 轨道倾角小于发射场纬度')
    # 计算窗口
    result_launch_window = []
    for theta in result_rotation:
        site_position_w = mr.vec_rotation(site_position, body_n, theta)
        # 排除方向错误的解
        site_arg = mv.angle_between_vectors(an_p, site_position_w)
        if (site_arg > np.pi / 2 and direction == 'NE' or \
            site_arg < np.pi / 2 and direction == 'SE'):
            continue
        time_difference = theta / (2 * np.pi) * period
        result_launch_window.append(time_difference + start_time)
    if len(result_launch_window) == 0:
        raise ValueError('无解, 没有可能的发射角度')
    if cloest == True:
        return min(result_launch_window)
    if not search:
        return [res for res in result_launch_window]

    # 计算最佳发射窗口
    min_phase       = min_phase
    max_phase       = max_phase
    start_time      = start_time
    end_time        = end_time
    # best_phase_angle = np.inf
    best_window     = result_launch_window[0]
    for i in range(len(result_launch_window)):
        theta       = result_rotation[i]
        launch_window_position = mr.vec_rotation(site_position, body_n, theta)
        # 计算搜索周期范围
        start_period    = (start_time - result_launch_window[i]) // period + 1
        start_period    = max(start_period, 0)
        start_period    = int(start_period)
        end_period      = (end_time - result_launch_window[i]) // period
        end_period      = int(end_period)
        for body_n in range(start_period, end_period):
            launch_window   = result_launch_window[i] + body_n * period
            target_position = orbit.propagate_to_epoch(launch_window).r_vec
            phase_angle     = mv.angle_between_vectors(launch_window_position, target_position)
            # 叉乘判断方向
            direction       = np.dot(np.cross(target_position, target_position - launch_window_position), orbit_plane_n) > 0
            if not direction:
                phase_angle = -phase_angle
            if min_phase < phase_angle < max_phase:
                return launch_window
            # if min_phase < phase_angle < best_phase_angle:
            #     best_phase_angle = phase_angle
            #     best_launch_window = launch_window

    return best_window