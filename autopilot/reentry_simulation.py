import numpy as np
import numpy.linalg as npl
from astropy import units as u
import threading
import time

from ..astro.orbit import Orbit
from ..astro.frame import BCIFrame
from ..spacecrafts import Spacecraft
from ..math import (
    scalar as ms, 
    vector as mv,
)
from ..utils import *


g0 = 9.80665
D9 = 1 / 9
D24 = 1 / 24

sim_params = {
    'dry_mass':             ...,
    'mass':                 ...,
    'min_throttle_cmp':     ...,  # 引擎支持的节流阀开度
    'max_throttle_cmp':     ...,
    'sim_throttle':         ...,  # 节流阀前馈
    'landing_asl':          ...,  # 着陆场海拔
    'vac_thrust':           ...,
    'vac_isp':              ...,
    'asl_thrust':           ...,
    'asl_isp':              ...,
    'suicide_check':        ...,  # 计算决断高度
}


class ReentrySimulation:
    # 再入大气轨迹预测
    def __init__(self,
                 spacecraft: Spacecraft,
                 orbit: Orbit,
                 params: dict):
        self.orbit              = orbit
        self.spacecraft         = spacecraft

        # 数值积分
        self._total_mass        = params['mass']
        self._dry_mass          = params['dry_mass']
        self._min_throttle      = params['min_throttle_cmp']
        self._max_throttle      = params['max_throttle_cmp']
        self._sim_throttle      = params['sim_throttle']
        self._landing_asl       = params['landing_asl']
        self._t                 = None
        self._mass              = None
        self._thrust            = None
        # bcbf左手系
        self._x                 = None
        self._v                 = None

        # krpc对象
        self._sc                = UTIL_CONN.space_center
        self._vessel            = spacecraft.vessel
        self._body              = orbit.attractor
        self._krpc_body         = self._sc.bodies.get(self._body.name)
        self._bcbf_ref          = self._krpc_body.reference_frame
        self._bci_ref           = self._krpc_body.non_rotating_reference_frame
        self._vessel_ref        = self._vessel.reference_frame
        self._flight            = self._vessel.flight(self._bcbf_ref)

        self._body_r            = self._krpc_body.equatorial_radius
        self._gm                = self._krpc_body.gravitational_parameter
        self._angular_v         = np.array(self._krpc_body.angular_velocity(self._bci_ref), dtype=np.float64)
        self._atmosphere_height = self._krpc_body.atmosphere_depth
        self._has_atmosphere    = self._atmosphere_height > 0

        # 计算推力
        self._vac_thrust        = params['vac_thrust']
        self._vac_isp           = params['vac_isp']
        self._asl_thrust        = params['asl_thrust']
        self._asl_isp           = params['asl_isp']
        if not self._has_atmosphere:
            self._pressure_curve    = UniformCurve(0, 0, 2, lambda x: 0)
        else:
            atm                     = int(self._atmosphere_height)
            self._pressure_curve    = UniformCurve(0, atm, atm // 1000, 
                                                   lambda x: self._krpc_body.pressure_at(x) / 101325)

        self._suicide_check_f   = params['suicide_check']
        self._suicide_index     = None
        self._trajectory        = None
        self.result             = SimulationResult()

        # 预分配
        self._suicide_check_x   = np.empty(3, dtype=np.float64)
        self._suicide_check_v   = np.empty(3, dtype=np.float64)
        self._v_local           = np.zeros(3, dtype=np.float64)

    def predict(self):
        # 处于大气层外的亚轨道, 传播到进入大气层后预测
        if not self._orbit_reenters():
            # 如果我们不在再入轨道上
            return

        if not self._has_atmosphere or \
            npl.norm(self.orbit.r_vec.to_value(u.m)) - self._body_r < self._atmosphere_height:
            # 已经进入大气或在无大气天体着陆
            reenter_orbit       = self.orbit
        else:
            # 传播到进入大气节省计算
            # FIXME: 在大气稀薄的情况下可能会出错
            reenter_r           = (self._body_r + self._atmosphere_height) * u.m
            reenter_orbit       = self.orbit.propagate_to_r(reenter_r, sign=False)

        bci_ref             = BCIFrame(reenter_orbit.attractor, reenter_orbit.epoch)
        self._t             = reenter_orbit.epoch.to_value(u.s)
        self._x             = reenter_orbit.r_vec.to_value(u.m)
        self._v             = reenter_orbit.v_vec.to_value(u.m / u.s)
        self._x             = bci_ref.transform_d_to_left_hand(self._x)
        self._v             = bci_ref.transform_d_to_left_hand(self._v)
        # 先转换速度
        self._v             = np.array(self._sc.transform_velocity(self._x, self._v, self._bci_ref, self._bcbf_ref))
        self._x             = np.array(self._sc.transform_position(self._x, self._bci_ref, self._bcbf_ref))
        self._trajectory    = ReentryTrajectory()

        self._free_fall()
        if self._suicide_check_f:
            if self._vac_thrust < 1:
                # TODO: 添加状态码
                self._suicide_index = 0
            else:
                self._suicide_altitude()
        self._trajectory.suicide_index = self._suicide_index
        self.result.update(self._trajectory)
        return self.result

    def decelerating(self, t, x, v, m, record=False):
        if record:
            if self._trajectory is not None:
                self._trajectory.clear()
            else:
                self._trajectory = ReentryTrajectory()
        return self._suicide_check(t, x, v, m, record=record)

    def _orbit_reenters(self):
        # TODO
        return True
        
    def _free_fall(self):
        dt = 1
        self._mass = self._total_mass
        while npl.norm(self._x) - self._body_r > self._landing_asl:
            self._trajectory.record(self._t, self._x, self._v)
            # print(npl.norm(self._x) - self._body_r)

            dx, dv, dt, next_dt = BS34_step(
                self._x, self._v, dt, self._total_acc_free_fall)

            # TODO: 开伞并减小步长逻辑
            # delta_x = npl.norm(self._x - self._x0)
            # if delta_x < 1000:
            #     next_dt = min(next_dt, 0.02)
            # elif delta_x < 5000:
            #     next_dt = min(next_dt, 0.5)
            # elif delta_x < 10000:
            #     next_dt = min(next_dt, 1)

            # TODO: 如果仍然能够逃逸大气层

            self._t += dt
            self._x += dx
            self._v += dv
            dt = next_dt
            # print(npl.norm(self._x) - self._body_r)

    def _suicide_altitude(self):
        # 二分搜索决断高度, 注意轨迹是不均匀采样的
        left = 0
        right = len(self._trajectory) - 1

        while right - left > 1:
            mid = (left + right) // 2
            t, x, v = self._trajectory.view[mid]
            _, res_x = self._suicide_check(t, x, v)
            if npl.norm(res_x) - self._body_r > self._landing_asl:
                # 仍然在着陆点上方
                left = mid
            else:
                right = mid

        # 此时决断高度在left与right之间, 二分搜索直到达到精度
        left_t, left_x, left_v = self._trajectory.view[left]
        # print('suicide alt sample:', npl.norm(left_x) - self._body_r)
        right_t, right_x, right_v = self._trajectory.view[right]
        while npl.norm(left_x - right_x) > 10:
            mid_t = (left_t + right_t) / 2
            mid_x = (left_x + right_x) / 2
            mid_v = (left_v + right_v) / 2

            _, res_x = self._suicide_check(mid_t, mid_x, mid_v)
            if npl.norm(res_x) - self._body_r > self._landing_asl:
                left_t, left_x, left_v = mid_t, mid_x, mid_v
            else:
                right_t, right_x, right_v = mid_t, mid_x, mid_v
        
        # print('suicide alt:', npl.norm(left_x) - self._body_r)
        self._trajectory._truncate(right)
        self._trajectory.record(left_t, left_x, left_v)
        self._suicide_index = right
        self._suicide_check(left_t, left_x, left_v, record=True)
        
    def _suicide_check(self, t, x, v, m=None, record=False):
        # 估算以固定节流减速下降至0的终点
        np.copyto(self._suicide_check_x, x)
        np.copyto(self._suicide_check_v, v)
        self._mass      = m if m is not None else self._total_mass
        dt              = 1

        max_thrust      = self._vac_thrust
        isp             = self._vac_isp
        self._thrust    = max_thrust * ms.lerp(self._min_throttle, self._max_throttle, self._sim_throttle)
        
        while True:
            if self._has_atmosphere:
                pressure        = self._pressure_curve.sample(npl.norm(self._suicide_check_x) - self._body_r)
                alpha           = 1 - min(max(pressure, 0), 1)
                max_thrust      = ms.lerp(self._asl_thrust, self._vac_thrust, alpha)
                isp             = ms.lerp(self._asl_thrust, self._vac_isp, alpha)
                self._thrust    = max_thrust * ms.lerp(self._min_throttle, self._max_throttle, self._sim_throttle)

            if record:
                self._trajectory.record(t, self._suicide_check_x, self._suicide_check_v)

            dx, dv, dt, next_dt = BS34_step(
                self._suicide_check_x, self._suicide_check_v, dt, 
                self._total_acc_decelerate
            )

            if npl.norm(dv) > npl.norm(self._suicide_check_v):  
                # 如果已经过冲, 估计减速到0的状态
                # 这个步骤似乎是没有必要的, 数值积分已经足够精确, 因此已经弃用
                # a = dv / dt  # 假设匀减速
                # dt = npl.norm(self._suicide_check_v) / npl.norm(a)
                # t += dt
                # self._suicide_check_x += self._suicide_check_v * dt + 0.5 * a * dt ** 2
                # self._suicide_check_v.fill(0)
                # if record:
                #     print('last state:')
                #     print('t, x, v:', self._t, self._suicide_check_x, self._suicide_check_v)
                #     print('dx, dv, dt:', dx, dv, dt)
                #     print('asl:', npl.norm(self._suicide_check_x) - self._body_r)
                #     self._trajectory.record(t, self._suicide_check_x, self._suicide_check_v)
                break

            self._mass -= self._thrust * dt / (isp * g0)
            self._mass = max(self._dry_mass, self._mass)  # TODO: 燃料不足
            t += dt
            self._suicide_check_x += dx
            self._suicide_check_v += dv
            dt = next_dt

            # print(round(npl.norm(self._suicide_check_v), 2), self._suicide_check_v.round(2))
        
        return t, self._suicide_check_x

    def _total_acc_free_fall(self, x, v):
        # 有空气动力而无推力的加速度, 直接调用krpc方法
        aerodynamic_acc     = self._get_aerodynamic_force(x, v) / self._mass
        g                   = -self._gm / np.dot(x, x) * mv.normalize(x)
        coriolis            = -2 * np.cross(self._angular_v, v)
        centrifugal         = -np.cross(self._angular_v, np.cross(self._angular_v, x))
        total_acc           = aerodynamic_acc + g + coriolis + centrifugal
        return total_acc

    def _total_acc_decelerate(self, x, v):
        base_acc            = self._total_acc_free_fall(x, v)
        thrust_acc          = -self._thrust * mv.normalize(v) / self._mass  # 假设推力始终逆向
        total_acc           = base_acc + thrust_acc
        return total_acc

    def _get_aerodynamic_force(self, x, v):
        # simlulate_aerodynamic_force_at方法受当前载具姿态影响
        # 这个方法在不同高度有一定误差, 确保减速平稳
        # 将速度矢量从机体系(0, -v, 0)转换到地面坐标系, 以便计算逆向姿态下的空气动力
        # 这个krpc方法是静态的, 所以我们无需担心轨道偏移的影响
        # 没有大气时, 我们不妨也进行数值积分, 因为无论如何我们需要离散化并寻找决断高度
        if not self._has_atmosphere:
            return np.zeros(3)
        self._v_local[1]    = -npl.norm(v)
        v_surface           = self._sc.transform_direction(self._v_local, self._vessel_ref, self._bcbf_ref)
        aero_force          = self._flight.simulate_aerodynamic_force_at(self._krpc_body, x, v_surface)
        aero_force          = -npl.norm(aero_force) * mv.normalize(v)
        return aero_force
    

class UniformCurve:
    def __init__(self, start, end, num, func):
        self.start = start
        self.end = end
        self.num = max(num, 2)
        self._step = (self.start - self.end) / (self.num - 1)
        self._xs = np.linspace(self.start, self.end, self.num)
        self._ys = np.zeros(num)
        for i in range(num):
            self._ys[i] = func(self._xs[i])
    
    def sample(self, x):
        if x <= self.start:
            return self._xs[0]
        if x >= self.end:
            return self._xs[-1]
        dx = x - self.start
        index = int(dx // self._step)
        alpha = dx % self._step / self._step
        return ms.lerp(self._ys[index], self._ys[index + 1], alpha)
        
        
class SimulationResult:
    def __init__(self):
        self._trajectory: ReentryTrajectory | None = None
        
    def update(self, trajectory):
        self._trajectory = trajectory

    def get(self):
        return self._trajectory


class ReentryTrajectory:
    def __init__(self):
        # FIXME: unit
        self._dtype = np.dtype([
            ('t', np.float64), 
            ('x', np.float64, (3,)), 
            ('v', np.float64, (3,))
        ])
        self._data = np.zeros(1000, dtype=self._dtype)
        self._size = 0
        self.suicide_index = None
        
    def __len__(self):
        return self._size

    @property
    def view(self):
        return self._data[:self._size]

    def get(self, index):
        return self._data[index].copy()

    @property
    def suicide_state(self):
        if self.suicide_index is None:
            return None
        return self.get(self.suicide_index)

    def record(self, t, x, v):
        if self._size >= len(self._data):
            temp = np.zeros(int(len(self._data) * 1.5), dtype=self._dtype)
            temp[:self._size] = self._data
            self._data = temp
        self._data[self._size][0] = t
        np.copyto(self._data[self._size][1], x)
        np.copyto(self._data[self._size][2], v)
        self._size += 1
        
    def clear(self):
        self._size = 0

    def _truncate(self, index):
        self._size = index

    def sample(self, t):
        data = self.view
        l = 0
        r = self._size - 1
        while r - l > 1:
            m = (l + r) // 2
            if data[m]['t'] < t:
                l = m + 1
            else:
                r = m
        l = data[l]
        r = data[r]
        if r['t'] - l['t'] < 1e-8:
            return l['x'], l['v']
        alpha = (t - l['t']) / (r['t'] - l['t'])
        x = ms.lerp(l['x'], r['x'], alpha)
        v = ms.lerp(l['v'], r['v'], alpha)
        return x, v


def BS34_step(x, v, dt, get_acc, min_dt=0.1, max_dt=100, tol=1e-5):
    repeat_with_smaller_step = True

    while repeat_with_smaller_step:

        dv1 = dt * get_acc(x, v)
        dx1 = dt * v

        dv2 = dt * get_acc(x + 0.5 * dx1, v + 0.5 * dv1)
        dx2 = dt * (v + 0.5 * dv1)

        dv3 = dt * get_acc(x + 0.75 * dx2, v + 0.75 * dv2)
        dx3 = dt * (v + 0.75 * dv2)

        dv4 = dt * get_acc(x + 2 * D9 * dx1 + 3 * D9 * dx2 + 4 * D9 * dx3,
                           v + 2 * D9 * dv1 + 3 * D9 * dv2 + 4 * D9 * dv3)
        
        dx = (2 * dx1 + 3 * dx2 + 4 * dx3) * D9
        dv = (2 * dv1 + 3 * dv2 + 4 * dv3) * D9

        zv = (7 * dv1 + 6 * dv2 + 8 * dv3 + 3 * dv4) * D24
        errorv = zv - dv
        error_mag = max(npl.norm(errorv), 1e-6)
        
        next_dt = dt * pow(tol / error_mag, 1/3) * 0.8

        if error_mag > tol and next_dt > min_dt:
            next_dt = ms.clip(next_dt, min_dt, max_dt)
            dt = next_dt
        else:
            next_dt = ms.clip(next_dt, min_dt, max_dt)
            repeat_with_smaller_step = False

    return dx, dv, dt, next_dt