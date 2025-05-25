import numpy as np
import numpy.linalg as npl
import time

from .reentry_simulation import ReentrySimulation
from ..astro.orbit import Orbit
from ..astro.maneuver import Maneuver
from ..astro.frame import BCIFrame
from ..spacecrafts import Spacecraft
from ..math import (scalar as ms, vector as mv, rotation as mr)
from ..utils import UTIL_CONN

g0 = 9.80665


def _mass_after_mnv(dv, m0, isp):
    return m0 / np.exp(dv / (isp * g0))

        
def deorbit(spacecraft: Spacecraft, landing_coord: tuple[float, float], deorbit_alt: float):
    # 降低轨道
    # FIXME: 如果已经进入了再入/降落轨道
    
    # 航天器首先滑行到近地点, 进行一次调相机动(可能的倾角机动)
    # 在目标时刻重返近地点时, 降低至圆形轨道
    # 在降低至圆形轨道后, 重新估计减速点火位置弥补误差
    # 滑行至减速点火点, 之后执行预计的减速点火

    # 首先按照圆形轨道和减速高度预估着陆轨迹
    # 反推滑行角度和重返近地点的目标时刻

    # bci右手系
    sc              = UTIL_CONN.space_center
    vessel          = sc.active_vessel
    body            = vessel.orbit.body
    body_r          = body.equatorial_radius
    body_period     = body.rotational_period
    bci_ref         = body.non_rotating_reference_frame
    bcbf_ref        = body.reference_frame
    # 注意此处orbit和landing_site时刻相同(t0)
    orbit           = Orbit.from_krpcv(vessel)
    body_h          = orbit.attractor.angular_velocity
    n               = mv.normalize(body_h)
    landing_site    = body.surface_position(*landing_coord, bci_ref)
    landing_site    = BCIFrame.transform_d_from_left_hand(landing_site)
    landing_asl     = npl.norm(landing_site) - body_r

    # 预测着陆轨迹后, 规划一个调相机动, 使得重返近地点时正好能够进入着陆轨迹
    # 这个方法无法确定一个合适的调相时间以精确规划着陆
    # revisit_epochs  = []
    # for t in phi_window:
    #     # 计算着陆窗口时着陆场的位置
    #     t               = t
    #     deorbit_epoch   = t - t_landing
    #     t_rotate        = t - t0
    #     theta_rotate    = t_rotate / body_period * 2 * np.pi
    #     # print(f'theta_rotate: {theta_rotate}')
    #     landing_site_f  = mr.vec_rotation(landing_site, n, theta_rotate)
    #     # 从近地点直到着陆场经过的角度
    #     angle_total     = mv.angle_between_vectors(x0, landing_site_f, h)
    #     # print(f'angle_total: {angle_total}')
    #     # 滑行至减速点火
    #     angle_waiting   = angle_total - angle_landing
    #     if angle_waiting < 0:
    #         angle_waiting += 2 * np.pi
    #     t_waiting       = angle_waiting / (np.pi * 2) * cir_period
    #     # print(f'gliding time: {t_waiting} angle: {angle_waiting}')
    #     # 重返近地点时目标时间 = 着陆窗口 - 着陆时间 - 滑行时间
    #     revisit_epoch   = deorbit_epoch - t_waiting
    #     if revisit_epoch - epoch_pe < cir_period:
    #         # 如果调相时间不足一个滑行周期, 等待之后的窗口
    #         # 使用滑行周期为基准的考虑是, 我们不希望调相轨道比滑行轨道更低(浪费delta_v)
    #         # 没有考虑当前ut已经错过着陆窗口的情况(虽然不太可能)
    #         iters           = (epoch_pe + cir_period - revisit_epoch) // body_period + 1
    #         revisit_epoch   += iters * body_period
    #     deorbit_epoch   = revisit_epoch + t_waiting
    #     revisit_epochs.append((revisit_epoch, deorbit_epoch))
    # revisit_epoch, deorbit_epoch = min(revisit_epochs, key=lambda x: x[0])
    # phase_mnv       = Maneuver.change_phase(orbit_pe, revisit_epoch, at_pe=True, safety_check=False, conserved=False)
    # orbit_phase     = phase_mnv.apply()
    # # print(f'change phase: {orbit_pe.epoch} dt: {orbit_pe.epoch - orbit.epoch}')
    # # print(phase_mnv)
    # orbit_phase     = orbit_phase.propagate_to_epoch(revisit_epoch)
    # circular_mnv    = Maneuver.change_apoapsis(orbit_phase, orbit_phase.pe, at_pe=False)
    # orbit_cir       = circular_mnv.apply()
    # # print(f'circularize: {orbit_phase.epoch} dt: {orbit_phase.epoch - orbit.epoch}')
    # # print(circular_mnv)
    # # 不真正执行减速点火, 而是在进入滑行轨道后重新估计弥补误差
    # # orbit_de        = orbit_cir.propagate_to_epoch(deorbit_epoch)
    # # deorbit_mnv     = Maneuver.change_apoapsis(orbit_de, deorbit_alt, at_pe=False)
    # # print(f'deorbit: {orbit_de.epoch} dt: {orbit_de.epoch - orbit.epoch}')
    # # print(deorbit_mnv)
    # mnv             = Maneuver.serial(orbit, [phase_mnv, circular_mnv])
    # mnv.to_krpcv(vessel)

    phase_flag = False

    # FIXME: 如果轨道倾角低于纬度
    # 估算圆轨道, 同时初次到达近地点
    circular_mnv    = Maneuver.change_apoapsis(orbit, orbit.pe)
    orbit_cir       = circular_mnv.apply()
    period_cir      = orbit_cir.period
    h_vec_cir       = orbit_cir.h_vec
    pos_pe          = orbit_cir.r_vec
    t_pe            = orbit_cir.epoch - orbit.epoch

    # 计算着陆点与轨道面重合窗口
    phi_window      = mr.solve_rotation_angle(landing_site, h_vec_cir, n, np.pi / 2)
    if phase_flag:
        phi_window  = min(phi_window)
    else:
        # 选距离更近的窗口
        phi_window  = min(phi_window, key=lambda x: abs((x + np.pi) % (2 * np.pi) - np.pi))
    landing_site_w  = mr.vec_rotation(landing_site, n, phi_window)

    # 总是在大约pi/2提前处进行倾角机动以提高效率
    # FIXME: 如果降落经过的角度angle_landing大于pi/2, 那么倾角改变后来不及降落
    mnv_dir         = np.cross(landing_site_w, h_vec_cir)
    # 近地点滑行至倾角机动经过的角度
    angle_waiting   = mv.angle_between_vectors(pos_pe, mnv_dir, h_vec_cir)
    t_waiting       = angle_waiting / (2 * np.pi) * period_cir

    if phase_flag:
        # 如果允许的话, 规划一个调相机动, 在着陆场与轨道面重合时达到倾角机动位置, 减少倾角机动消耗
        # 当前轨道 -> 近地点 -> 调相 -> 重返近地点 -> 圆化 -> 倾角(重合窗口) -> 降轨
        # 重返时刻 = 当前时刻 + 重合时间 - 近地点到倾角时间 - 当前位置到近地点时间
        # FIXME: 如果调相时间过短
        t_window        = phi_window / (2 * np.pi) * body_period
        # 提前t_waiting到达近地点
        epoch_revisit   = orbit.epoch + t_window - t_waiting
        orbit_pe        = orbit.propagate_to_nu(0)
        phase_mnv       = Maneuver.change_phase(orbit_pe, epoch_revisit)
        orbit_phase     = phase_mnv.apply()
        # 传播到近地点并圆化轨道
        orbit_phase     = orbit_phase.propagate_to_epoch(epoch_revisit)
        circular_mnv    = Maneuver.change_apoapsis(orbit_phase, orbit.pe, immediate=True)
        orbit_cir       = circular_mnv.apply()
        # FIXME: 不真正执行后续任务而是将这个任务返回
        # 从近地点滑行到倾角机动点
        orbit_approx    = orbit_cir.propagate(t_waiting)
        h_dir_approx    = orbit_approx.h_vec
        dv              = phase_mnv.get_total_cost() + circular_mnv.get_total_cost()
        # 从初始轨道到倾角机动的时间
        t_to_mnv        = t_window
    else:
        # 如果不进行调相机动, 构建重合轨道, 即经过倾角机动和当前着陆场位置的轨道, 作为倾角机动估计
        # 当前轨道 -> 近地点 -> 圆化 -> 倾角 -> 降轨
        h_dir_approx    = np.cross(mnv_dir, landing_site)
        r_vec_new       = orbit_cir.r * mv.normalize(mnv_dir)
        v_dir           = np.cross(h_dir_approx, r_vec_new)
        v_vec_new       = (npl.norm(h_vec_cir) / npl.norm(r_vec_new)) * mv.normalize(v_dir)
        orbit_approx    = Orbit.from_rv(orbit_cir.attractor, r_vec_new, v_vec_new, orbit_cir.epoch)
        # 预估这次机动的dv
        inc_mnv         = Maneuver.match_plane(orbit_cir, orbit_approx, conserved=True)
        dv              = inc_mnv.get_total_cost() + circular_mnv.get_total_cost()
        t_to_mnv        = t_waiting + t_pe


    # FIXME: 这里大致估计了一下改变倾角后滑行到减速位置经过的角度theta - angle_landing
    # 弥补非惯性力不一致的问题
    # NOTE: 也许可以根据着陆场位置估算? 当前落月初始误差不超过1km
    orbit_approx    = orbit_approx.propagate_to_nu(orbit_approx.nu + np.pi / 4)
    # 假设降低轨道用于降落估计
    deorbit_mnv     = Maneuver.change_apoapsis(orbit_approx, deorbit_alt, immediate=True)
    orbit_de        = deorbit_mnv.apply()

    # 估计所有机动消耗的质量
    dv              += deorbit_mnv.get_total_cost()
    dv              = dv
    sim_mass        = _mass_after_mnv(dv, vessel.mass, vessel.specific_impulse_at(0))
    # FIXME
    sim_mass = vessel.mass

    t0              = orbit_de.epoch
    x0              = orbit_de.r_vec
    
    sim_params = {
        'dry_mass':             vessel.dry_mass,
        'mass':                 sim_mass,
        'min_throttle_cmp':     0.374866,
        'max_throttle_cmp':     1,
        'sim_throttle':         0.9,
        'landing_asl':          landing_asl,
        'vac_thrust':           vessel.max_thrust_at(0),
        'vac_isp':              vessel.specific_impulse_at(0),
        'asl_thrust':           vessel.max_thrust_at(1),
        'asl_isp':              vessel.specific_impulse_at(1),
        'suicide_check':        True
    }

    print(f'fuel consumed: {vessel.mass - sim_mass}')
    print(f'landing_site: {landing_site}')
    print(f'landing asl: {landing_asl}')
    print('running...')

    # 这里的模拟是有误差的, 在不同的位置受到的非惯性力不一致, 每1deg真近点角带来的误差大约1km
    sim             = ReentrySimulation(spacecraft, orbit_de, sim_params)
    t_sim_s         = sc.ut
    res             = sim.predict()
    traj            = res.get()
    tf, xf, _       = traj.view[-1]
    t_landing       = tf - t0
    t_sim           = sc.ut - t_sim_s
    # FIXME: 这里的转换在实时系统中是有误差的, 考虑建立BCBF参考系
    xf              = sc.transform_position(xf, bcbf_ref, bci_ref)
    xf              = BCIFrame.transform_d_from_left_hand(xf)
    # 补偿着陆期间的自转
    xf              = mr.vec_rotation(xf, n, (t_landing - t_sim) / body_period * (2 * np.pi))

    # 着陆轨迹并不是在一个平面内的, 而是向自转方向略微弯曲, 估计由此带来的误差
    # 估计自转方向的额外误差, FIXME: orbit_de的位置并不准确
    angle_proj      = mr.solve_rotation_angle(xf, h_dir_approx, n, np.pi / 2)
    angle_proj      = [(x + np.pi) % (2 * np.pi) - np.pi for x in angle_proj]
    angle_proj      = min(angle_proj, key=lambda x: abs(x))
    xf_proj         = mr.vec_rotation(xf, n, angle_proj)
    # 轨道平面内的落地夹角
    angle_landing   = mv.angle_between_vectors(x0, xf_proj, h_dir_approx)
    angle_landing_f = mv.angle_between_vectors(x0, xf, h_dir_approx)

    print(f'landing time: {t_landing} angle: {np.rad2deg(angle_landing)} real: {np.rad2deg(angle_landing_f)}')
    print(f'angle proj: {np.rad2deg(angle_proj)}')

    # 数值求解:
    # 求着陆场的旋转角phi, 旋转后着陆场与当前航天器夹角theta, 满足
    # (phi - angle_proj) / (2 * pi) * body_period 
    # = (theta - angle_landing) / (2 * pi) * orbit_period + t_to_mnv + t_landing
    # 即旋转后两者重合
    # a = orbit_period / body_period
    # b = (-angle_landing * orbit_period + (t_to_mnv + t_landing) * (2 * pi)) / body_period + angle_proj
    a               = period_cir / body_period
    b               = (-angle_landing * period_cir + (t_to_mnv + t_landing) * (2 * np.pi)) / body_period + angle_proj
    # 以着陆到当前位置的theta作为猜测解
    guess           = a * mv.angle_between_vectors(mnv_dir, landing_site) + b
    phi             = _find_rotation_angle(landing_site, mnv_dir, n, a, b, guess)
    theta           = (phi - b) / a
    # 瞄准位置
    landing_site_t  = mr.vec_rotation(landing_site, n, phi + angle_proj)

    print(f'window: {np.rad2deg(phi_window)}')
    print(f'guess: {np.rad2deg(guess)}')
    print(f'phi: {np.rad2deg(phi)} theta: {np.rad2deg(theta)}')
    print(f'theta error: {mv.angle_between_vectors(landing_site_t, mnv_dir) - theta}')
    print(f'waiting: {np.rad2deg(angle_waiting)}deg {t_waiting}s')
    print(f'gliding: {np.rad2deg(theta - angle_landing)}deg')

    # 滑行至降轨机动
    t_gliding       = (theta - angle_landing) / (2 * np.pi) * period_cir

    orbit_inc       = orbit_cir.propagate_to_epoch(orbit.epoch + (t_to_mnv - 60))
    h_dir           = np.cross(mnv_dir, landing_site_t)
    # 这里由于mnv_dir已经是提前pi/2的位置, 所以不需要判断方向
    r_vec_new       = orbit_inc.r * mv.normalize(mnv_dir)
    v_dir           = np.cross(h_dir, r_vec_new)
    v_vec_new       = (npl.norm(h_vec_cir) / npl.norm(r_vec_new)) * mv.normalize(v_dir)
    orbit_target    = Orbit.from_rv(orbit_inc.attractor, r_vec_new, v_vec_new, orbit_inc.epoch)
    inc_mnv         = Maneuver.match_plane(orbit_inc, orbit_target, closest=True, conserved=True)
    
    # 降轨
    orbit_de        = inc_mnv.apply()
    orbit_de        = orbit_de.propagate(t_gliding)
    deorbit_mnv     = Maneuver.change_apoapsis(orbit_de, deorbit_alt, immediate=True)

    if phase_flag:
        mnv = Maneuver.serial(orbit, [phase_mnv, circular_mnv, inc_mnv, deorbit_mnv])
    else:
        mnv = Maneuver.serial(orbit, [circular_mnv, inc_mnv, deorbit_mnv])
        
    mnv.to_krpcv(vessel)

    orbit_inc = orbit_inc.propagate(60)
    print(f'inc mnv pos diff: {np.rad2deg(mv.angle_between_vectors(mnv_dir, orbit_inc.r_vec, orbit_inc.h_vec))}')
    orbit_f = mnv.apply()
    print(mnv)
    print(f'deorbit epoch diff: {orbit_f.epoch - t_gliding - t_to_mnv - orbit.epoch}')
    print(f'landing site diff: {np.rad2deg(mv.angle_between_vectors(landing_site_t, orbit_f.h_vec))}')
    input('ready to cheat')
    print(orbit_f.cheat())


def _find_rotation_angle(v, u, n, a, b, guess):
    v /= np.linalg.norm(v)
    u /= np.linalg.norm(u)
    n /= np.linalg.norm(n)

    A = np.dot(u, v)
    B = np.dot(u, np.cross(n, v))
    C = np.dot(n, v) * np.dot(u, n)
    D = A - C
    M = np.linalg.norm(u) * np.linalg.norm(v)

    def f(phi):
        return D*np.cos(phi) + B*np.sin(phi) + C \
             - M*np.cos((phi - b)/a)

    def df(phi):
        return -D*np.sin(phi) + B*np.cos(phi) \
             + (M/a)*np.sin((phi - b)/a)

    phi = guess
    max_iter = 100
    tol = 1e-10

    for i in range(max_iter):
        fval = f(phi)
        dfval = df(phi)
        if abs(dfval) < 1e-14:
            break
        delta = fval/dfval
        phi -= delta
        if abs(delta) < tol:
            break

    phi = (phi + np.pi) % (2*np.pi) - np.pi
    return phi
    

def landing(spacecraft: Spacecraft, landing_coord: tuple):
    # bcbf左手系
    sc              =  UTIL_CONN.space_center
    vessel          =  sc.active_vessel
    body            =  vessel.orbit.body
    orbit           =  Orbit.from_krpcv(vessel)

    bcbf_ref        =  body.reference_frame
    local_ref       =  vessel.reference_frame
    body_r          =  body.equatorial_radius
    g_surface       =  body.gravitational_parameter / body_r ** 2
    landing_site    =  np.array(body.surface_position(*landing_coord, bcbf_ref), dtype=np.float64)
    landing_asl     =  npl.norm(landing_site) - body_r
    flight          =  vessel.flight(bcbf_ref)
    control         =  vessel.control
    ap              =  vessel.auto_pilot

    debug_line      = True
    debug_line_len  = 10

    final_height    = 500
    max_tilt        = np.deg2rad(10)
    mass            = vessel.mass
    vac_thrust      = vessel.max_thrust_at(0)
    vac_isp         = vessel.specific_impulse_at(0)
    asl_thrust      = vessel.max_thrust_at(1)
    asl_isp         = vessel.specific_impulse_at(1)
    sim_throttle    = 0.9
    min_throttle_cmp    = 0.374866
    max_throttle_cmp    = 1
    min_throttle_ctrl   = 0.001
    max_throttle_ctrl   = 1

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
    # 将着陆点强制修改为预测落点, 用于测试
    # landing_site = x_final
    # landing_asl = npl.norm(landing_site) - body_r
    x_e             = landing_site - x_final
    # FIXME: 如果发现初始误差较大应该修正点火位置, 当前只是粗略估计
    # 估计修正速度
    v_correct       = npl.norm(x_e) / (t_final - t_suicide)
    # 估计修正可用加速度
    a_correct       = vac_thrust / mass * (max_throttle_ctrl - sim_throttle)
    t_correct       = v_correct / a_correct
    t_suicide       -= t_correct
    x_suicide, _    = traj.sample(t_suicide)
    suicide_alt     = npl.norm(x_suicide) - body_r

    x_i = mv.normalize(landing_site)
    x_e_ver = np.dot(x_e, x_i) * x_i
    x_e_hor = x_e - x_e_ver
    h_i = mv.normalize(np.cross(x_suicide, v_suicide))
    x_e_nor = np.dot(x_e_hor, h_i) * h_i
    x_e_par = x_e_hor - x_e_nor
    print(f'landing time: {t_final - t_0} angle: {np.rad2deg(mv.angle_between_vectors(x_final, x_0))}')
    print(f'error: {npl.norm(x_e)}')
    print(f'error_ver: {npl.norm(x_e_ver)} error_par: {npl.norm(x_e_par)} error_nor: {npl.norm(x_e_nor)}')
    print(f'correct dt: {t_correct} corrected suicide alt: {suicide_alt}')

    if debug_line:
        print('plotting...')
        lines, target_mark, thrust_line, error_line, sample_line = \
            _add_lines(traj, landing_site, bcbf_ref, local_ref, debug_line_len)

    print('Gliding to deceleration point')
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

    ap.reference_frame = bcbf_ref
    ap.target_direction = -np.array(vel_stream())
    ap.engage()

    # 动力减速
    print('Decelerating')
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

        if dt < 1e-6:
            continue
        if npl.norm(x) - body_r < landing_asl + final_height:
            break

        t_final, x_final = sim.decelerating(t, x, v, m)
        x_e         = landing_site - x_final
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
            # update_lines(traj, lines)
            temp = sc.transform_direction(T, bcbf_ref, local_ref)
            thrust_line.end = temp / npl.norm(temp) * 10
            temp = sc.transform_direction(T_e, bcbf_ref, local_ref)
            error_line.end = temp / npl.norm(temp) * 10
            temp = sc.transform_position(landing_site, bcbf_ref, local_ref)
            sample_line.end = temp
            target_mark.thickness = np.clip(npl.norm(x - landing_site) / 100, 1, 100)

    print('Final descent')
    control.gear    = True
    alt_stream      = UTIL_CONN.add_stream(getattr, flight, 'surface_altitude')
    vessel_h        = 5  # TODO
    
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
        if alt < 0.5:
            ap.target_direction = x_i
            control.throttle = 0
            break

        max_acc     = max_throttle_cmp * (u / m) - g_surface
        min_acc     = min_throttle_cmp * (u / m) - g_surface
        mid_acc     = ms.lerp(max(0, min_acc), max_acc, 0.5)

        v_norm      = np.sqrt(2 * mid_acc * alt)
        v_t         = -v_norm * x_i

        if alt > final_height * 0.5:
            # 如果大于最终高度, 进行落点修正
            x_e         = landing_site - x
            x_e_ver     = np.dot(x_e, x_i) * x_i
            x_e_hor     = x_e - x_e_ver
            dx_e        = (x_e_hor - x_e_prev) / dt
            v_t         += 0.5 * x_e_hor + 0.1 * dx_e
            x_e_prev    = x_e_hor

        v_t         *= min(npl.norm(v_t) / 20 + 0.1, 1)  # 快速衰减
        v_t         = mv.conic_clamp(v_t, v, 0, v_norm, max_tilt, prograde=True)
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

