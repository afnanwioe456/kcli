from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from .scalar import OrbitBase
from .utils import orbit_launch_window
from ..core.kepler import *
from ..core.lagrange import rv2rv_delta_t
from ..body import *
from ..utils import get_ut

if TYPE_CHECKING:
    from krpc.services.spacecenter import Vessel, Orbit as KRPCOrbit


class Orbit(OrbitBase):
    def propagate(self, t):
        """传播时间t后的轨道

        Args:
            t (float): 传播时间.

        Returns:
            Orbit: 传播后的轨道
        """
        if self.e < 1e-10:
            nu = self.nu + t / self.period * (2 * np.pi)
            return self.propagate_to_nu(nu)
        if self.e < 1:
            dt = t % self.period
        else:
            dt = t
        r_vec = self.r_vec
        v_vec = self.v_vec
        k = self.attractor.mu
        r1_vec, v1_vec = rv2rv_delta_t(r_vec, v_vec, dt, k)
        return self.from_rv(
            self.attractor,
            r1_vec,
            v1_vec,
            self.epoch + t
        )

    def propagate_to_epoch(self, epoch):
        """传播到指定时间的轨道

        Args:
            epoch (float): 时刻.

        Returns:
            Orbit: 传播后的轨道
        """
        dt = epoch - self.epoch
        return self.propagate(dt)

    def propagate_to_nu(self, nu, prograde=True, M=0):
        """传播到指定真近点角的轨道

        Args:
            nu (float): 真近点角
            prograde (bool): 严格按正向传播
            M (int): 传播经过的完整周期

        Returns:
            Orbit: 传播后的轨道
        """
        orb = Orbit.from_coe(self.attractor, self.a, self.e, self.inc, 
                             self.raan, self.argp, nu, self.epoch)
        dt = orb.delta_t - self.delta_t
        if prograde and dt < 0:
            M += 1
        if M != 0:
            if self.e > 1:
                raise ValueError()
            else:
                dt += M * self.period
        orb._epoch = dt + orb._epoch
        return orb

    def propagate_to_r(self, r, sign=True, prograde=True, M=0):
        nu = r2nu(r, self.h, self.e, self.attractor.mu, sign)
        return self.propagate_to_nu(nu, prograde, M)

    def is_safe_before(self, epoch):
        if epoch < self.epoch:
            raise ValueError(f'epoch {epoch} is smaller than orbit epoch {self.epoch}')
        if epoch - self.epoch >= self.period:
            # 如果周期数大于1
            return self.is_safe()
        orb = self.propagate_to_epoch(epoch)
        if orb.nu < self.nu:
            # 如果经过了近星点
            return self.is_safe()
        if 2 * np.pi - orb.nu < self.nu:
            # 如果epoch处更接近近星点
            return orb.is_safe()
        return self.is_safe(self.nu)
        
    @staticmethod
    def from_coe(attractor, a, e, inc, raan, argp, nu, epoch):
        """从轨道根数创建Orbit对象

        Args:
            attractor (Body): 中心天体.
            a (float): 半长轴.
            e (float): 离心率.
            inc (float): 轨道倾角.
            raan (float): 升交点经度.
            argp (float): 近地点辐角.
            nu (float): 真近点角.
            epoch (float): KSPRO历元时刻, 自1951-01-01 00:00:00以来的秒数.

        Returns:
            Orbit: 轨道.
        """
        # FIXME: 没有判断双曲线nu是否合法
        orb = Orbit()
        orb._assign_coe(attractor, a, e, inc, raan, argp, nu, epoch)
        return orb

    @staticmethod
    def from_rv(attractor, r_vec, v_vec, epoch):
        """从rv状态向量创建Orbit对象

        Args:
            attractor (Body): 中心天体.
            r_vec (ndarray): 位置矢量.
            v_vec (ndarray): 速度矢量.
            epoch (float): KSPRO历元时刻, 自1951-01-01 00:00:00以来的秒数.

        Returns:
            Orbit: 轨道.
        """
        orb = Orbit()
        orb._assign_rv(attractor, r_vec, v_vec, epoch)
        return orb

    @classmethod
    def circular(cls, attractor, r_vec, h_i, epoch):
        """圆轨道

        Args:
            attractor (Body): 中心天体
            r_vec (ndarray): 位置矢量
            h_i (ndarray): 角动量方向
            epoch (float): KSPRO历元时刻, 自1951-01-01 00:00:00以来的秒数.

        Returns:
            Orbit: 圆轨道
        """
        r = np.linalg.norm(r_vec)
        h = np.sqrt(attractor.k * r)
        v_i = np.cross(h_i, r_vec)
        v_i = v_i / np.linalg.norm(v_i)
        v_vec = h / r * v_i
        return cls.from_rv(attractor, r_vec, v_vec, epoch)
    
    @classmethod
    def from_krpcv(cls, vessel: Vessel):
        """从krpc Vessel对象创建Orbit对象

        Args:
            vessel (Vessel): krpc Vessel
        """
        return cls.from_krpcorb(vessel.orbit)


    @classmethod
    def from_krpcorb(cls, orbit: KRPCOrbit, epoch: float | None = None):
        """从krpc Orbit对象创建Orbit对象

        Args:
            orbit (Orbit): krpc Orbit
            ut (float | None): epoch
        """
        epoch = get_ut() if epoch is None else epoch
        nu = orbit.true_anomaly_at_ut(epoch)
        nu = nu % (2 * np.pi)
        return cls.from_coe(
            Body.get_or_create(orbit.body.name),
            orbit.semi_major_axis,
            orbit.eccentricity,
            orbit.inclination,
            orbit.longitude_of_ascending_node,
            orbit.argument_of_periapsis,
            nu,
            epoch,
        )

    def launch_window(self, 
                      site_p: np.ndarray, 
                      direction: str = 'SE', 
                      cloest: bool = False, 
                      search: bool = True,
                      min_phase: float = None, 
                      max_phase: float = None,
                      start_time: float = None, 
                      end_time: float = None) -> list[float] | float:
        """发射场向轨道发射的时刻

        Args:
            site_p (ndarray): 发射场位置矢量, 惯性系
            direction (str): 发射方向. Defaults to 'SE'.
            cloest (bool): 返回最近的窗口. Defaults to False.
            search (bool): 搜索最佳窗口. Defaults to True.
            min_phase (float, optional): 最小相位差. Defaults to None.
            max_phase (float, optional): 最大相位差. Defaults to None.
            start_time (float, optional): 搜索开始时间. Defaults to None.
            end_time (float, optional): 搜索终止时间. Defaults to None.

        Returns:
            list[float]: 发射窗口
        """
        if min_phase is None:
            min_phase = np.deg2rad(10)
        if max_phase is None:
            min_phase = np.deg2rad(30)
        if start_time is None or end_time is None:
            start_time = self.epoch
            end_time = self.epoch + self.attractor.rotational_period * 10
        ut = orbit_launch_window(
            self, site_p, direction, cloest, search,
            min_phase, max_phase, start_time, end_time
        )
        return ut

    def cheat(self, ut=None):
        if ut is None:
            ut = get_ut()
        orb = self.propagate_to_epoch(ut)
        nu = orb.nu
        e = orb.e
        E = nu2E(nu, e)
        Me = E2Me(E, e)
        return (f'SMA: {orb.a}\n'
                f'ECC: {orb.e}\n'
                f'INC: {np.rad2deg(orb.inc)}\n'
                f'MNA: {Me}\n'
                f'LAN: {np.rad2deg(orb.raan)}\n'
                f'ARG: {np.rad2deg(orb.argp)}')

    def _to_dict(self):
        return {
            'attractor':    self.attractor.name,
            'a':            self.a,
            'e':            self.e,
            'inc':          self.inc,
            'raan':         self.raan,
            'argp':         self.argp,
            'nu':           self.nu,
            'epoch':        self.epoch,
        }

    @classmethod
    def _from_dict(cls, data):
        from ..body import Body
        return cls.from_coe(
            attractor   = Body.get_or_create(data['attractor']),
            a           = data['a'],
            e           = data['e'],
            inc         = data['inc'],
            raan        = data['raan'],
            argp        = data['argp'],
            nu          = data['nu'],
            epoch       = data['epoch'],
        )
        
        