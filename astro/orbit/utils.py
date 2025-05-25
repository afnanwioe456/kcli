from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

from ...math import (
    rotation as mr,
    vector as mv
)

if TYPE_CHECKING:
    from .create import Orbit


def orbit_launch_window(orbit: Orbit, 
                        site_position: np.ndarray, 
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
    if start_time > end_time or end_time < orbit.epoch:
        raise ValueError()
    period = orbit.attractor.rotational_period
    # bci右手系
    # 获得目标轨道面的法向量
    target_position = orbit.r_vec
    target_raan = orbit.raan
    an_p = np.array([np.cos(target_raan), np.sin(target_raan), 0])  # 升交点方向
    orbit_plane_normal = orbit.h_vec
    n = orbit.attractor.angular_velocity
    n = mv.normalize(n)  # 数值稳定
    # 计算重合时的旋转角
    result_rotation = mr.solve_rotation_angle(site_position, orbit_plane_normal, n, np.pi / 2)
    if result_rotation is None:
        raise ValueError('无解, 轨道倾角小于发射场纬度')
    # 计算窗口
    result_launch_window = []
    for i in range(len(result_rotation)):
        theta = result_rotation[i]
        proj_result_p = mr.vec_rotation(site_position, n, theta)
        proj_result_p[2] = 0
        # 排除方向错误的解
        site_arg = mv.angle_between_vectors(an_p, proj_result_p)
        if (site_arg > np.pi / 2 and direction == 'NE' or \
            site_arg < np.pi / 2 and direction == 'SE'):
            result_rotation.pop(i)
            continue
        time_difference = theta * period
        result_launch_window.append(time_difference + orbit.epoch)
    if len(result_launch_window) == 0:
        raise ValueError('无解, 没有可能的发射角度')
    if cloest == True:
        return min(result_launch_window)
    if not search:
        return [res for res in result_launch_window]

    # 计算最佳发射窗口
    min_phase = min_phase
    max_phase = max_phase
    start_time = start_time
    end_time = end_time
    # best_phase_angle = np.inf
    best_launch_window = result_launch_window[0]
    for i in range(len(result_launch_window)):
        theta = result_rotation[i]
        launch_window_position = mr.vec_rotation(site_position, n, theta)
        # 计算搜索周期范围
        start_period = (start_time - result_launch_window[i]) // period + 1
        start_period = max(start_period, 0)
        end_period = (end_time - result_launch_window[i]) // period
        for n in range(start_period, end_period):
            launch_window = result_launch_window[i] + n * period
            target_position = orbit.propagate_to_epoch(launch_window).r_vec
            phase_angle = mv.angle_between_vectors(launch_window_position, target_position)
            # 叉乘判断方向
            direction = np.dot(np.cross(target_position, target_position - launch_window_position), orbit_plane_normal) > 0
            if not direction:
                phase_angle = -phase_angle
            if min_phase < phase_angle < max_phase:
                return launch_window
            # if min_phase < phase_angle < best_phase_angle:
            #     best_phase_angle = phase_angle
            #     best_launch_window = launch_window

    return best_launch_window