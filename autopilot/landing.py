import numpy as np
import numpy.linalg as npl
import time

from .reentry_simulation import ReentrySimulation
from ..astro.orbit import Orbit
from ..spacecrafts import Spacecraft
from ..math import (scalar as ms, vector as mv)
from ..utils import UTIL_CONN, time_wrap

# min_throttle lanyue: 0.14559906264796402  KTDU-425: 0.374866

def landing(spacecraft: Spacecraft, landing_coord: tuple):
    # TODO:
    final_height    = 200
    vessel_h        = 2
    v_touchdown     = 0.5
    debug_line      = True
    debug_line_len  = 20
    sim_throttle    = 0.9
    min_throttle_cmp    = 0.14559906264796402
    max_throttle_cmp    = 1
    min_throttle_ctrl   = 0.001
    max_throttle_ctrl   = 1

    # bcbf左手系
    sc              = UTIL_CONN.space_center
    vessel          = sc.active_vessel
    body            = vessel.orbit.body
    orbit           = Orbit.from_krpcv(vessel)

    bcbf_ref        = body.reference_frame
    local_ref       = vessel.reference_frame
    body_r          = body.equatorial_radius
    g_surface       = body.gravitational_parameter / body_r ** 2
    landing_site    = np.array(body.surface_position(*landing_coord, bcbf_ref), dtype=np.float64)
    _ls_norm        = npl.norm(landing_site)
    landing_site_de = landing_site * (_ls_norm + final_height) / _ls_norm
    landing_asl     = npl.norm(landing_site_de) - body_r
    flight          = vessel.flight(bcbf_ref)
    control         = vessel.control
    ap              = vessel.auto_pilot

    max_tilt        = np.deg2rad(10)
    mass            = vessel.mass
    vac_thrust      = vessel.max_thrust_at(0)
    vac_isp         = vessel.specific_impulse_at(0)
    asl_thrust      = vessel.max_thrust_at(1)
    asl_isp         = vessel.specific_impulse_at(1)

    sim_params = {
        'dry_mass':             vessel.dry_mass,
        'mass':                 mass,
        'min_throttle_cmp':     min_throttle_cmp,
        'max_throttle_cmp':     max_throttle_cmp,
        'sim_throttle':         sim_throttle,
        'landing_asl':          landing_asl,
        'vac_thrust':           vac_thrust,
        'vac_isp':              vac_isp,
        'asl_thrust':           asl_thrust,
        'asl_isp':              asl_isp,
        'suicide_check':        True
    }

    print('running...')
    sim             = ReentrySimulation(spacecraft, orbit, sim_params)
    res             = sim.predict()
    traj            = res.get()

    # 计算偏差
    t_0, x_0, _ = traj.view[0]
    t_final, x_final, _ = traj.view[-1]
    t_suicide, x_suicide, v_suicide = traj.suicide_state
    x_e             = landing_site_de - x_final
    # FIXME: 如果发现初始误差较大应该修正点火位置, 当前只是粗略估计
    # 估计修正可用加速度
    a_correct       = vac_thrust / mass * (max_throttle_cmp - min_throttle_cmp) / 2
    t_correct       = t_final - np.sqrt(2 * a_correct * npl.norm(x_e))
    t_suicide       = min(t_suicide, t_correct)
    x_suicide, _    = traj.sample(t_suicide)
    suicide_alt     = npl.norm(x_suicide) - body_r

    x_i = mv.normalize(landing_site_de)
    x_e_ver = np.dot(x_e, x_i) * x_i
    x_e_hor = x_e - x_e_ver
    h_i = mv.normalize(np.cross(x_suicide, v_suicide))
    x_e_nor = np.dot(x_e_hor, h_i) * h_i
    x_e_par = x_e_hor - x_e_nor
    print(f'landing time: {t_final - t_0} angle: {np.rad2deg(mv.angle_between_vectors(x_final, x_0))}')
    print(f'error: {npl.norm(x_e)}')
    print(f'error_ver: {npl.norm(x_e_ver)} error_par: {npl.norm(x_e_par)} error_nor: {npl.norm(x_e_nor)}')
    print(f'corrected dt: {t_suicide - t_0} corrected suicide alt: {suicide_alt}')

    if debug_line:
        print('plotting...')
        lines, target_mark, thrust_line, error_line, sample_line = \
            _add_lines(traj, landing_site, bcbf_ref, local_ref, debug_line_len)

    print('Gliding to deceleration point')
    time_wrap(t_suicide - 30)
    vessel.control.rcs = True
    ap.reference_frame = bcbf_ref
    ap.target_direction = -np.array(flight.velocity)
    ap.engage()

    mean_altitude_call = UTIL_CONN.get_call(getattr, flight, 'mean_altitude')
    expr = UTIL_CONN.krpc.Expression.less_than(
        UTIL_CONN.krpc.Expression.call(mean_altitude_call),
        UTIL_CONN.krpc.Expression.constant_double(suicide_alt))
    event = UTIL_CONN.krpc.add_event(expr)
    with event.condition:
        event.wait()

    ut_stream       = UTIL_CONN.add_stream(getattr, sc, 'ut')
    pos_stream      = UTIL_CONN.add_stream(getattr, flight, 'center_of_mass')
    vel_stream      = UTIL_CONN.add_stream(getattr, flight, 'velocity')
    mass_stream     = UTIL_CONN.add_stream(getattr, vessel, 'mass')
    thrust_stream   = UTIL_CONN.add_stream(getattr, vessel, 'max_thrust')
    alt_stream      = UTIL_CONN.add_stream(getattr, flight, 'surface_altitude')

    # 动力减速
    print('Decelerating')
    final_height += 10
    control.throttle = 0.001
    base_thrust_factor = ms.lerp(min_throttle_cmp, max_throttle_cmp, sim_throttle)

    v_e_prev        = 0
    t_prev          = ut_stream()
    Kp, Kd          = 0.2, 0.1

    while True:
        time.sleep(0.1)
        t           = ut_stream()
        dt          = t - t_prev
        t_prev      = t
        x           = np.array(pos_stream(), dtype=np.float64)
        v           = np.array(vel_stream(), dtype=np.float64)
        m           = mass_stream()
        u           = thrust_stream()
        alt         = alt_stream() - vessel_h

        if dt < 1e-6:
            continue
        if alt < final_height:
            break

        t_final, x_final = sim.decelerating(t, x, v, m, record=debug_line)
        x_e         = landing_site_de - x_final
        t_total     = t_final - t
        v_e         = x_e / t_total
        dv_e        = (v_e - v_e_prev) / dt
        v_e_prev    = v_e

        # 开环
        # a           = v_e / dt * 0.9  # 略微衰减避免计算时间差异过大导致过冲
        # 闭环
        a           = Kp * v_e + Kd * dv_e
        T_e         = a * m
        T_0         = -base_thrust_factor * u * mv.normalize(v)
        T           = T_0 + T_e
        T           = mv.conic_clamp(T, -v, 0.001, u, max_tilt, prograde=True)
        T_norm      = npl.norm(T)
        direction   = T / T_norm
        # FIXME: 节流阀-推力非线性(如LMDE)
        throttle    = (T_norm / u - min_throttle_cmp) / (max_throttle_cmp - min_throttle_cmp)
        throttle    = np.clip(throttle, min_throttle_ctrl, max_throttle_ctrl)
        print(f'throttle: {throttle:.2f} dt: {dt:.2f} '
              f'error: {npl.norm(x_e):.2f}m height: {npl.norm(x_final) - body_r - landing_asl:.2f}')

        ap.target_direction = direction
        control.throttle = throttle

        if debug_line:
            traj = sim.result.get()
            _update_lines(traj, lines)
            temp = sc.transform_direction(T, bcbf_ref, local_ref)
            thrust_line.end = temp / npl.norm(temp) * 10
            temp = sc.transform_direction(T_e, bcbf_ref, local_ref)
            error_line.end = temp / npl.norm(temp) * 10
            temp = sc.transform_position(landing_site, bcbf_ref, local_ref)
            sample_line.end = temp
            target_mark.thickness = np.clip(npl.norm(x - landing_site) / 100, 1, 100)

    print('Final descent')
    control.gear    = True
    v_e_prev        = 0
    x_e_prev        = 0
    t_prev          = ut_stream()
    while True:
        time.sleep(0.1)
        t           = ut_stream()
        x           = np.array(pos_stream(), dtype=np.float64)
        v           = np.array(vel_stream(), dtype=np.float64)
        u           = thrust_stream()
        m           = mass_stream()
        alt         = alt_stream() - vessel_h

        dt          = t - t_prev
        x_i         = mv.normalize(x)

        if dt < 1e-6:
            continue
        if alt < 1:
            ap.target_direction = x_i
            control.throttle = 0
            break

        max_acc     = max_throttle_cmp * (u / m) - g_surface
        min_acc     = min_throttle_cmp * (u / m) - g_surface
        mid_acc     = ms.lerp(max(0, min_acc), max_acc, 0.5)

        v_norm      = np.sqrt(2 * mid_acc * alt)
        v_norm      = max(v_norm, v_touchdown)
        v_t         = -v_norm * x_i

        # 进行水平落点修正
        x_e         = landing_site - x
        x_e_ver     = np.dot(x_e, x_i) * x_i
        x_e_hor     = x_e - x_e_ver
        dx_e        = (x_e_hor - x_e_prev) / dt
        v_t         += 0.6 * x_e_hor + 0.1 * dx_e
        x_e_prev    = x_e_hor

        v_t         *= min(npl.norm(v_t) / 20 + 0.1, 1)  # 快速衰减
        v_t         = mv.conic_clamp(v_t, v, v_touchdown, v_norm, max_tilt, prograde=True)
        v_e         = v_t - v
        dv_e        = (v_e - v_e_prev) / dt
        v_e_prev    = v_e
        t_prev      = t

        print(f'alt: {alt:.2f} vt: {npl.norm(v_t):.2f} '
              f'v: {npl.norm(v_t - v):.2f} error: {npl.norm(landing_site - x):.2f}')

        a           = v_e * 0.6 + dv_e * 0.1
        T_e         = m * a
        T_0         = m * mid_acc * x_i
        T           = T_0 + T_e
        T           = mv.conic_clamp(T, -v, 0.001, u, max_tilt, prograde=True)
        T_norm      = npl.norm(T)
        direction   = T / T_norm
        throttle    = (T_norm / u - min_throttle_cmp) / (max_throttle_cmp - min_throttle_cmp)
        throttle    = np.clip(throttle, min_throttle_ctrl, max_throttle_ctrl)

        ap.target_direction = direction
        control.throttle = throttle

        if debug_line:
            temp = sc.transform_direction(T, bcbf_ref, local_ref)
            thrust_line.end = temp / npl.norm(temp) * 10
            temp = sc.transform_direction(T_e, bcbf_ref, local_ref)
            error_line.end = temp / npl.norm(temp) * 10
            temp = sc.transform_position(landing_site, bcbf_ref, local_ref)
            sample_line.end = temp

    print('Landed')

    ut_stream.remove()
    pos_stream.remove()
    vel_stream.remove()
    mass_stream.remove()
    thrust_stream.remove()
    alt_stream.remove()

    # 沿路径下降
    # i               = traj.suicide_index + 1
    # view            = traj.view
    # x_e_prev        = 0
    # v_e_prev        = 0
    # t_prev          = ut_stream()
    # Kp, Kd          = 0.2, 0.05

    # while True:
    #     time.sleep(0.1)
    #     t = ut_stream()
    #     dt = t - t_prev
    #     if dt < 1e-6:
    #         continue
    #     x = np.array(pos_stream(), dtype=np.float64)
    #     v = np.array(vel_stream(), dtype=np.float64)
    #     m = mass_stream()
    #     u = thrust_stream()

    #     res = sim.decelerating()
    #     traj = res.get()
    #     t_final, x_final, _ = traj.view[-1]
    #     x_e = landing_site - x_final
    #     print(round(npl.norm(x_e), 2))

    #     if npl.norm(x) - body_r < landing_asl + final_height:
    #         break

    #     while view[i]['t'] < t:
    #         i += 1
    #     i = min(i, len(view) - 1)
    #     t0, x0, v0 = view[i-1]
    #     t1, x1, v1 = view[i  ]
    #     alpha = (t - t0) / (t1 - t0)
    #     x_t = ms.lerp(x0, x1, alpha)
    #     v_t = ms.lerp(v0, v1, alpha)
    #     x_e = x_t - x
    #     v_e = v_t - v

    #     dx_e = (x_e - x_e_prev) / dt
    #     v_e += Kp * x_e + Kd * dx_e
    #     dv_e = (v_e - v_e_prev) / dt
    #     a = Kp * v_e + Kd * dv_e
        
    #     x_e_prev = x_e
    #     v_e_prev = v_e
    #     t_prev = t

    #     T_e = a * m
    #     T_0 = -base_thrust_factor * u * mv.normalize(v)
    #     T = T_0 + T_e
    #     T = mv.conic_clamp(T, -v, 0, u, max_tilt)
    #     T_norm = npl.norm(T)
    #     direction = T / T_norm
    #     throttle = (T_norm / u - min_throttle) / (1 - min_throttle)
    #     throttle = max(0.001, throttle)
    #     ap.target_direction = direction
    #     control.throttle = throttle

    #     if debug_line:
    #         temp = sc.transform_direction(T, bcbf_ref, local_ref)
    #         thrust_line.end = temp / npl.norm(temp) * 10
    #         # print(temp, throttle, npl.norm(T_e))
    #         temp = sc.transform_direction(T_e, bcbf_ref, local_ref)
    #         error_line.end = temp / npl.norm(temp) * 10
    #         temp = sc.transform_position(landing_site, bcbf_ref, local_ref)
    #         sample_line.end = temp

def _plot_target_polygon(v, ref):
    v = np.array(v, dtype=np.float64)
    v_norm = np.linalg.norm(v)
    v_unit = v / v_norm

    if abs(v_unit[0]) < 0.99:
        a = np.array([1, 0, 0])
    else:
        a = np.array([0, 1, 0])

    u0 = np.cross(v_unit, a)
    u0 /= np.linalg.norm(u0)
    w = np.cross(v_unit, u0)
    w /= np.linalg.norm(w)

    angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    perps = [(np.cos(theta) * u0 + np.sin(theta) * w) * 1000 for theta in angles]
    vertices = [
        v, v + perps[0], v + perps[1],
        v, v + perps[3], v + perps[4],
        v, v + perps[6], v + perps[7],
    ]
    poly = UTIL_CONN.drawing.add_polygon(vertices, ref)
    poly.color = (1.0, 0.647, 0)
    poly.thickness = 1
    return poly
    

def _add_lines(trajectory, landing_site, ref, local_ref, max_len):
    n = min(len(trajectory), max_len)
    # n = len(trajectory)
    traj_lines = [UTIL_CONN.drawing.add_line((0, 0, 0), (0, 0, 0), ref) for _ in range(n)]
    target_polygon = _plot_target_polygon(landing_site, ref)
    l1 = UTIL_CONN.drawing.add_line((0, 0, 0), (0, 0, 0), local_ref)
    l1.color = (0, 0, 1)
    l2 = UTIL_CONN.drawing.add_line((0, 0, 0), (0, 0, 0), local_ref)
    l2.color = (1, 0, 0)
    l3 = UTIL_CONN.drawing.add_line((0, 0, 0), (0, 0, 0), local_ref)
    l3.color = (1, 1, 0)
    return traj_lines, target_polygon, l1, l2, l3
        

def _update_lines(trajectory, lines):
    n = min(len(trajectory), len(lines))
    trajectory = trajectory.view
    for i in range(n - 1):
        lines[i].start = tuple(trajectory[i]['x'])
        lines[i].end = tuple(trajectory[i + 1]['x'])
    for i in range(n, len(lines)):
        lines[i].start = (0, 0, 0)
        lines[i].end = (0, 0, 0)

