import time

from utils import *
import numpy as np
import sympy as sp
import math
WENCHANG_LAUNCH_PAD_LATITUDE = 19.613726150307052
WENCHANG_LAUNCH_PAD_LONGITUDE = 110.9553275138089


def angle_between_vectors(vector_1, vector_2):
    return np.degrees(np.arccos(np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))))


def angle_between_vectors_on_xOz(vector1, vector2):
    """
    xoz平面上的两个向量v1，v2，视线向y轴负方向，返回v2向v1逆时针扫过的弧度
    """
    cross_product = np.cross(vector1, vector2)
    dot_product = np.dot(vector1, vector2)
    norm_1 = np.linalg.norm(vector1)
    norm_2 = np.linalg.norm(vector2)

    angle_cos = dot_product / (norm_1 * norm_2)
    angle_arc = np.arccos(angle_cos)

    if cross_product[1] < 0:
        angle_arc = 2 * np.pi - angle_arc

    return angle_arc


def orbit_launch_window(target: Vessel, direction='ANY', start_period=0, end_period=30):
    """
    发射到target轨道面的发射窗口
    -> ([(最近两次发射窗口, 发射方向) ...], start-end周期之间目标方向的最佳发射窗口)
    """
    body = target.orbit.body
    period = body.rotational_period
    # 使用地心惯性参考系
    ref_frame = body.non_rotating_reference_frame
    # 获取变量：目标真近点角，发射场位置向量(x, y, z)，ut
    ut = get_ut()
    site_position = body.surface_position(WENCHANG_LAUNCH_PAD_LATITUDE, WENCHANG_LAUNCH_PAD_LONGITUDE, ref_frame)
    target_orbit = target.orbit
    target_argument_of_periapsis = target_orbit.argument_of_periapsis
    # 获得目标轨道面的法向量
    target_position = target_orbit.position_at(ut, ref_frame)
    target_velocity = target.velocity(ref_frame)
    orbit_plane_normal = np.cross(target_position, target_velocity)
    # TODO: 轨道倾角小于发射场纬度时
    # 计算重合时的位置向量(x1, y, z1) (x2, y, z2)
    a, b, c = orbit_plane_normal
    x = sp.Symbol('x')
    y = site_position[1]
    z = sp.Symbol('z')
    result_value = sp.solve([a * x + b * y + c * z,
                             x ** 2 + y ** 2 + z ** 2 - np.linalg.norm(site_position) ** 2],
                            [x, z])
    # 向赤道面投影并计算向量夹角，由周期确定重合倒计时
    result_launch_window = []
    for i in list(range(len(result_value))):
        phase_difference = angle_between_vectors_on_xOz((float(result_value[i][0]), 0, float(result_value[i][1])),
                                                        (site_position[0], 0, site_position[2]))
        time_difference = abs(phase_difference) / np.pi * period / 2
        result_launch_window.append(ut + time_difference)
    # 计算发射窗口的发射方向
    an_true_anomaly = 2 * np.pi - target_argument_of_periapsis
    an_position = target_orbit.position_at(target_orbit.ut_at_true_anomaly(an_true_anomaly), ref_frame)
    result_launch_direction = []
    for i in list(range(len(result_value))):
        launch_window_position = (float(result_value[i][0]), y, float(result_value[i][1]))
        if angle_between_vectors(an_position, launch_window_position) < 90:
            result_launch_direction.append((result_launch_window[i], 'NE'))
        else:
            result_launch_direction.append((result_launch_window[i], 'SE'))

    print('最近的发射窗口：')
    for i in list(range(len(result_launch_window))):
        print(sec_to_date(result_launch_window[i]), result_launch_direction[i])

    # 计算最佳发射窗口

    best_phase_angle = 180
    best_launch_window = result_launch_window[0]
    best_launch_direction = result_launch_direction[0]
    for i in list(range(len(result_value))):
        if direction != 'ANY' and direction != result_launch_direction[i][1]:
            continue
        launch_window = result_launch_window[i] + start_period * period
        launch_window_position = (float(result_value[i][0]), y, float(result_value[i][1]))
        for j in range(start_period, end_period):
            target_position = target_orbit.position_at(launch_window, ref_frame)
            phase_angle = angle_between_vectors(launch_window_position, target_position)
            target_position_delta = target_orbit.position_at(launch_window + 10, ref_frame)
            phase_angle_delta = angle_between_vectors(launch_window_position, target_position_delta)
            if phase_angle_delta < phase_angle:
                phase_angle = -phase_angle
            print(sec_to_date(launch_window), result_launch_direction[i][1], phase_angle)
            # TODO: 设置相位差参数
            if 40 < phase_angle < best_phase_angle:
                best_phase_angle = phase_angle
                best_launch_window = launch_window
                best_launch_direction = result_launch_direction[i]
            launch_window += period
    print(f'{start_period}至{end_period}太阳日内最佳发射窗口：\n'
          f'{sec_to_date(best_launch_window)}')

    return result_launch_direction, (best_launch_window, best_launch_direction)


if __name__ == '__main__':
    conn = krpc.connect()
    direction = 'SE'
    result = orbit_launch_window(direction=direction, start_period=0, end_period=30, target=get_vessel_by_name('KSS'))
    best_launch_window = result[1][0]
    best_launch_direction = result[1][1]

    from launch import Soyuz2Launch
    from task.tasks import TaskQueue
    from command import ChatMsg
    from task.tasks import Tasks
    from spacecrafts import KSS
    import threading

    task_queue = TaskQueue()
    event = threading.Event()
    event.set()
    msg = ChatMsg('1', 'launch_to_rendezvous', '1', '1', 0)
    tasks = Tasks(msg, 0, 2, task_queue)

    target_vessel = KSS.vessel
    target_name = KSS.name
    if best_launch_direction == 'SE':
        inc = np.degrees(target_vessel.orbit.inclination)
    else:
        inc = np.degrees(-target_vessel.orbit.inclination)
    print(inc)
    # tasks.submit(Soyuz2Launch(tasks, inclination=inc, start_time=best_launch_window - 360))
    # tasks.do()

