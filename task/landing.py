from __future__ import annotations
import numpy as np
import numpy.linalg as npl
from time import sleep
from krpc.services.spacecenter import SASMode

from .tasks import Task
from .maneuver import ExecuteNode
from ..astro.orbit import Orbit
from ..astro.maneuver import Maneuver
from ..astro.constants import G0
from ..astro.frame import BCIFrame
from ..autopilot.reentry_simulation import ReentrySimulation
from ..autopilot.landing import landing
from ..math import (vector as mv, rotation as mr)
from ..utils import *


if TYPE_CHECKING:
    from .tasks import Tasks
    from ..spacecrafts import Spacecraft
    from ..astro.body import Body


__all__ = [
    'GlideLanding',
    'LandingMnv',
    'Landing'
]


class GlideLanding(Task):
    def __init__(self, 
                 spacecraft: Spacecraft, 
                 tasks: Tasks, 
                 start_time: float = -1, 
                 duration: float = 1800, 
                 importance: int = 3,
                 submit_next: bool = True):
        super().__init__(spacecraft, tasks, start_time, duration, importance, submit_next)

    @property
    def description(self):
        return (f'{self.spacecraft.name} -> 无动力着陆'
                f'\t预计执行时: {sec_to_date(self.start_time)}')

    @logging_around
    def start(self):
        if not self._conn_setup():
            return
        krpc_orb = self.vessel.orbit
        orbv = Orbit.from_krpcorb(krpc_orb)
        while orbv.pe > orbv.attractor.atmosphere_height:
            # 如果航天器此时还未转移到环绕要着陆的天体轨道上
            # FIXME: 判断过于简略, 如果已经进入大气层的情况
            krpc_orb = krpc_orb.next_orbit
            if krpc_orb is None:
                raise ValueError(f'{self.spacecraft.name} not on a reentry orbit.')
            orbv = Orbit.from_krpcorb(krpc_orb)
        orb_reentry = orbv.propagate_to_r(orbv.attractor.r + orbv.attractor.atmosphere_height, sign=False)
        time_wrap(orb_reentry.epoch)
        self.vessel = self.sc.active_vessel
        while not get_parts_in_stage_by_type(
            self.vessel, 
            'parachute', 
            self.vessel.control.current_stage):
            self.vessel.control.activate_next_stage()
            sleep(1)
            self.vessel = self.sc.active_vessel
        self.vessel.control.rcs = True
        self.vessel.control.sas = True
        sleep(0.1)
        self.vessel.control.sas_mode = SASMode.retrograde
        self.sc.physics_warp_factor = 3
        recover_stream = self.conn.add_stream(getattr, self.vessel, 'recoverable')
        mass_stream = self.conn.add_stream(getattr, self.vessel, 'mass')
        while not recover_stream() or mass_stream > 0:
            sleep(1)
        if self.vessel.recoverable:
            self.vessel.recover()
        dummy_roll_out()


class LandingMnv(Task):
    def __init__(self, 
                 spacecraft: Spacecraft, 
                 tasks: Tasks, 
                 body: Body,
                 landing_coord: tuple[float, float],
                 deorbit_alt: float,
                 start_time: float = -1, 
                 duration: float = 600, 
                 importance: int = 3,
                 submit_next: bool = True):
        """着陆机动规划"""
        super().__init__(spacecraft, tasks, start_time, duration, importance, submit_next)
        self.body = body
        self.landing_coord = landing_coord
        self.deorbit_alt = deorbit_alt
        # FIXME
        self._phase_flag = True

    @property
    def description(self):
        return (f'{self.spacecraft.name} -> 着陆规划 -> {self.body.name}\n'
                f'\t坐标: {self.landing_coord}\n'
                f'\t预计执行时: {sec_to_date(self.start_time)}')

    @logging_around
    def start(self):
        if not self._conn_setup():
            return

        # FIXME: 如果已经进入了再入/降落轨道
        
        # 航天器首先滑行到近地点, 进行一次调相机动(可能的倾角机动)
        # 在目标时刻重返近地点时, 降低至圆形轨道
        # 在降低至圆形轨道后, 重新估计减速点火位置弥补误差
        # 滑行至减速点火点, 之后执行预计的减速点火

        # 首先按照圆形轨道和减速高度预估着陆轨迹
        # 反推滑行角度和重返近地点的目标时刻

        # FIXME
        final_height    = 200
        sim_throttle    = 0.9
        min_throttle_cmp    = 0.14559906264796402
        max_throttle_cmp    = 1

        # bci右手系
        sc              = self.sc
        vessel          = self.vessel
        body            = vessel.orbit.body
        bci_ref         = body.non_rotating_reference_frame
        bcbf_ref        = body.reference_frame
        # 注意此处orbit和landing_site时刻相同(t0)
        orbit           = Orbit.from_krpcv(vessel)
        attractor       = orbit.attractor
        body_h          = attractor.angular_velocity
        n               = mv.normalize(body_h)
        body_r          = attractor.r
        body_period     = attractor.rotational_period
        landing_site    = attractor.surface_position(self.landing_coord, orbit.epoch)
        # FIXME: 将着陆场修正到最终下降处, 需要去除硬编码
        _ls_norm        = npl.norm(landing_site)
        landing_site    = landing_site * (_ls_norm + final_height) / _ls_norm
        landing_asl     = npl.norm(landing_site) - body_r
        t_sim_s         = sc.ut

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
        if self._phase_flag:
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

        # 如果不进行调相机动, 构建重合轨道, 即经过倾角机动和当前着陆场位置的轨道, 作为倾角机动估计
        # 当前轨道 -> 近地点 -> 圆化 -> 倾角 -> 降轨
        h_dir_approx    = np.cross(mnv_dir, landing_site)
        r_vec_new       = orbit_cir.r * mv.normalize(mnv_dir)
        v_dir           = np.cross(h_dir_approx, r_vec_new)
        v_vec_new       = (npl.norm(h_vec_cir) / npl.norm(r_vec_new)) * mv.normalize(v_dir)
        orbit_approx    = Orbit.from_rv(orbit_cir.attractor, r_vec_new, v_vec_new, orbit_cir.epoch)
        # 预估这次机动的dv
        inc_mnv         = Maneuver.match_plane(orbit_cir, orbit_approx, conserved=True)
        dv_inc          = inc_mnv.get_total_cost()
        dv_circular     = circular_mnv.get_total_cost()
        t_to_mnv        = t_waiting + t_pe

        if self._phase_flag:
            self._phase_flag = False
            # 如果允许的话, 规划一个调相机动, 在着陆场与轨道面重合时达到倾角机动位置, 减少倾角机动消耗
            # 当前轨道 -> 近地点 -> 调相 -> 重返近地点 -> 圆化 -> 倾角(重合窗口) -> 降轨
            # 重返时刻 = 当前时刻 + 重合时间 - 近地点到倾角时间 - 当前位置到近地点时间
            t_window        = phi_window / (2 * np.pi) * body_period
            # 提前t_waiting到达近地点
            epoch_revisit   = orbit.epoch + t_window - t_waiting
            orbit_pe        = orbit.propagate_to_nu(0)
            phase_mnv       = Maneuver.change_phase(orbit_pe, epoch_revisit)
            dv_phase        = phase_mnv.get_total_cost()
            # 如果调相时间过短, 则直接进行倾角机动
            # 如果调相机动比倾角机动更低效, 则直接进行倾角机动
            if phase_mnv is not None and dv_phase < dv_inc:
                orbit_phase     = phase_mnv.apply()
                # 传播到近地点并圆化轨道
                orbit_phase     = orbit_phase.propagate_to_epoch(epoch_revisit)
                circular_mnv    = Maneuver.change_apoapsis(orbit_phase, orbit.pe, immediate=True)
                # 不真正执行后续任务而是在调相圆化后重新执行此任务
                mnv             = Maneuver.serial(orbit, [phase_mnv, circular_mnv])
                nodes           = mnv.to_krpcv(vessel)
                tasks           = [ExecuteNode.from_node(self.spacecraft, node, self.tasks) for node in nodes]
                tasks.append(self)
                self.tasks.submit_nowait(tasks)
                return

        # FIXME: 这里大致估计了一下改变倾角后滑行到减速位置经过的角度theta - angle_landing
        # 弥补非惯性力不一致的问题
        # NOTE: 也许可以根据着陆场位置估算? 当前落月初始误差不超过1km
        orbit_approx    = orbit_approx.propagate_to_nu(orbit_approx.nu + np.pi / 4)
        # 假设降低轨道用于降落估计
        deorbit_mnv     = Maneuver.change_apoapsis(orbit_approx, self.deorbit_alt, immediate=True)
        orbit_de        = deorbit_mnv.apply()

        # 估计所有机动消耗的质量
        dv_deorbit      = deorbit_mnv.get_total_cost()
        dv              = dv_inc + dv_circular + dv_deorbit
        sim_mass        = self._mass_after_mnv(dv, vessel.mass, vessel.specific_impulse_at(0))

        t0              = orbit_de.epoch
        x0              = orbit_de.r_vec
        
        sim_params = {
            'dry_mass':             vessel.dry_mass,
            'mass':                 sim_mass,
            'min_throttle_cmp':     min_throttle_cmp,
            'max_throttle_cmp':     max_throttle_cmp,
            'sim_throttle':         sim_throttle,
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
        sim             = ReentrySimulation(self.spacecraft, orbit_de, sim_params)
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
        phi             = self._find_rotation_angle(landing_site, mnv_dir, n, a, b, guess)
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
        deorbit_mnv     = Maneuver.change_apoapsis(orbit_de, self.deorbit_alt, immediate=True)

        mnv             = Maneuver.serial(orbit, [circular_mnv, inc_mnv, deorbit_mnv])
        nodes           = mnv.to_krpcv(vessel)

        # orbit_inc = orbit_inc.propagate(60)
        # print(f'inc mnv pos diff: {np.rad2deg(mv.angle_between_vectors(mnv_dir, orbit_inc.r_vec, orbit_inc.h_vec))}')
        # orbit_f = mnv.apply()
        # print(mnv)
        # print(f'deorbit epoch diff: {orbit_f.epoch - t_gliding - t_to_mnv - orbit.epoch}')
        # print(f'landing site diff: {np.rad2deg(mv.angle_between_vectors(landing_site_t, orbit_f.h_vec))}')
        # input('ready to cheat')
        # print(orbit_f.cheat())

        if self.submit_next:
            self._submit_next_task(nodes)
        
    def _submit_next_task(self, nodes):
        next_task: list[Task] = []
        for n in nodes:
            task = ExecuteNode.from_node(self.spacecraft, n, self.tasks, importance=8)
            next_task.append(task)
        self.tasks.submit_nowait(next_task)

    def _find_rotation_angle(self, v, u, n, a, b, guess):
        v /= npl.norm(v)
        u /= npl.norm(u)
        n /= npl.norm(n)

        A = np.dot(u, v)
        B = np.dot(u, np.cross(n, v))
        C = np.dot(n, v) * np.dot(u, n)
        D = A - C
        M = npl.norm(u) * npl.norm(v)

        def f(phi):
            return D*np.cos(phi) + B*np.sin(phi) + C \
                - M*np.cos((phi - b)/a)

        def df(phi):
            return -D*np.sin(phi) + B*np.cos(phi) \
                + (M/a)*np.sin((phi - b)/a)

        phi = guess
        max_iter = 100
        tol = 1e-8

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

    def _mass_after_mnv(self, dv, m0, isp):
        return m0 / np.exp(dv / (isp * G0))

    def _to_dict(self):
        dic =  {
            'body_name': self.body.name,
            'landing_coord': self.landing_coord,
            'deorbit_alt': self.deorbit_alt,
            'phase_flag': self._phase_flag,
        }
        return super()._to_dict() | dic
    
    @classmethod
    def _from_dict(cls, data, tasks):
        from ..spacecrafts import Spacecraft
        from ..astro.body import Body
        ret = cls(
            spacecraft      = Spacecraft.get(data['spacecraft_name']),
            tasks           = tasks,
            body            = Body.get_or_create(data['body_name']),
            landing_coord   = tuple(data['landing_coord']),
            deorbit_alt     = data['deorbit_alt'],
            start_time      = data['start_time'],
            duration        = data['duration'],
            importance      = data['importance'],
            submit_next     = data['submit_next'],
        )
        ret._phase_flag = data['phase_flag']
        return ret
        

class Landing(Task):
    def __init__(self, 
                 spacecraft: Spacecraft, 
                 tasks: Tasks, 
                 body: Body,
                 landing_coord: tuple[float, float],
                 start_time: float = -1, 
                 duration: float = 1800, 
                 importance: int = 9,
                 submit_next: bool = True):
        """着陆"""
        super().__init__(spacecraft, tasks, start_time, duration, importance, submit_next)
        self.body = body
        self.landing_coord = landing_coord

    @property
    def description(self):
        return (f'{self.spacecraft.name} -> 着陆 -> {self.body.name}\n'
                f'\t坐标: {self.landing_coord}\n'
                f'\t预计执行时: {sec_to_date(self.start_time)}')

    @logging_around
    def start(self):
        if not self._conn_setup():
            return
        landing(self.spacecraft, self.landing_coord)
        
    def _to_dict(self):
        dic =  {
            'body_name': self.body.name,
            'landing_coord': self.landing_coord,
        }
        return super()._to_dict() | dic
    
    @classmethod
    def _from_dict(cls, data, tasks):
        from ..spacecrafts import Spacecraft
        from ..astro.body import Body
        ret = cls(
            spacecraft      = Spacecraft.get(data['spacecraft_name']),
            tasks           = tasks,
            body            = Body.get_or_create(data['body_name']),
            landing_coord   = tuple(data['landing_coord']),
            start_time      = data['start_time'],
            duration        = data['duration'],
            importance      = data['importance'],
            submit_next     = data['submit_next'],
        )
        return ret