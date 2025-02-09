from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from astropy import units as u

from .scalar import OrbitBase
from .utils import orbit_launch_window
from ..core.kepler import *
from ..core.lagrange import rv2rv_delta_t
from ..body import *
from ..utils import get_ut

if TYPE_CHECKING:
    from krpc.services.spacecenter import Vessel, Orbit as KRPCOrbit


ATTRACTOR_DIC = {
    'Earth': KSP_Earth,
    'Moon': KSP_Moon,
}

class Orbit(OrbitBase):
    @u.quantity_input(t=u.s)
    def propagate(self, t, tol=1e-8, max_iter=100):
        """传播时间t后的轨道

        Args:
            t (Quantity): 传播时间.
            tol (float, optional): 误差. Defaults to 1e-8.
            max_iter (int, optional): 最大迭代数. Defaults to 100.

        Returns:
            Orbit: 传播后的轨道
        """
        dt = t.to_value(u.s)
        r_vec = self.r_vec.to_value(u.km)
        v_vec = self.v_vec.to_value(u.km / u.s)
        k = self.attractor.k.to_value(u.km ** 3 / u.s ** 2)
        r1_vec, v1_vec = rv2rv_delta_t(r_vec, v_vec, dt, k, tol, max_iter)
        return self.from_rv(
            self.attractor,
            r1_vec * u.km,
            v1_vec * u.km / u.s,
            self.epoch + t
        )

    @u.quantity_input(epoch=u.s)
    def propagate_to_epoch(self, epoch, tol=1e-8, max_iter=100):
        """传播到指定时间的轨道

        Args:
            epoch (Quantity): 时刻.
            tol (float, optional): 误差. Defaults to 1e-8.
            max_iter (int, optional): 最大迭代数. Defaults to 100.

        Returns:
            Orbit: 传播后的轨道
        """
        dt = epoch - self.epoch
        return self.propagate(dt, tol, max_iter)

    @u.quantity_input(nu=u.rad)
    def propagate_to_nu(self, nu, prograde=False, M=0):
        """传播到指定真近点角的轨道

        Args:
            nu (Quantity): 真近点角
            prograde (bool): 严格按正向传播
            M (int): 传播经过的完整周期

        Returns:
            Orbit: 传播后的轨道
        """
        if M != 0:
            prograde = True
        orb = Orbit.from_coe(self.attractor, self.a, self.e, self.inc, 
                             self.raan, self.argp, nu, self.epoch)
        dt = orb.delta_t - self.delta_t
        if prograde and dt < 0:
            dt += self.period
        dt += M * self.period
        orb._epoch = dt + orb._epoch  # 不要使用+=
        return orb

    @u.quantity_input(r=u.km)
    def propagate_to_r(self, r, sign=True, prograde=True, M=0):
        nu = r2nu(
            r.to_value(u.km),
            self.h.to_value(u.km ** 2 / u.s),
            self.e.to_value(u.one),
            self.attractor.k.to_value(u.km ** 3 / u.s ** 2),
            sign
        ) * u.rad
        return self.propagate_to_nu(nu, prograde, M)

    @u.quantity_input(nu=u.s)
    def is_safe_before(self, epoch):
        if epoch < self.epoch:
            raise ValueError(f'epoch {epoch} is smaller than orbit epoch {self.epoch}')
        if epoch >= self.period:
            return self.is_safe()
        orb = self.propagate_to_epoch(epoch)
        if orb.nu <= self.nu:
            return self.is_safe()
        p = np.pi * u.rad
        if (abs(p - orb.nu) < abs(p - self.nu)):
            return self.is_safe(self.nu)
        return self.is_safe(orb.nu)

    @staticmethod
    @u.quantity_input(
        a=u.km,
        ecc=u.one,
        inc=u.rad,
        raan=u.rad,
        argp=u.rad,
        nu=u.rad,
        epoch=u.s,
    )
    def from_coe(attractor, a, e, inc, raan, argp, nu, epoch):
        """从轨道根数创建Orbit对象

        Args:
            attractor (Body): 中心天体.
            a (Quantity): 半长轴.
            e (Quantity): 离心率.
            inc (Quantity): 轨道倾角.
            raan (Quantity): 升交点经度.
            argp (Quantity): 近地点辐角.
            nu (Quantity): 真近点角.
            epoch (Quantity): KSPRO历元时刻, 自1951-01-01 00:00:00以来的秒数.

        Returns:
            Orbit: 轨道.
        """
        orb = Orbit()
        orb._assign_coe(attractor, a, e, inc, raan, argp, nu, epoch)
        return orb

    @staticmethod
    @u.quantity_input(
        r_vec=u.km,
        v_vec=u.km/u.s,
        epoch=u.s,
    )
    def from_rv(attractor, r_vec, v_vec, epoch):
        """从rv状态向量创建Orbit对象

        Args:
            attractor (Body): 中心天体.
            r_vec (ndarray[Quantity]): 位置矢量.
            v_vec (ndarray[Quantity]): 速度矢量.
            epoch (Quantity): KSPRO历元时刻, 自1951-01-01 00:00:00以来的秒数.

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
            r_vec (Quantity): 位置矢量
            h_i (Quantity): 角动量方向
            epoch (Quantity): KSPRO历元时刻, 自1951-01-01 00:00:00以来的秒数.

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
    def from_krpcorb(cls, orbit: KRPCOrbit, ut=None):
        """从krpc Orbit对象创建Orbit对象

        Args:
            orbit (Orbit): krpc Orbit
            ut (Quantity | None): epoch
        """
        ut = get_ut() if ut is None else ut.to_value(u.s)
        nu = orbit.true_anomaly_at_ut(ut)
        if nu < 0:
            nu += 2 * np.pi
        return cls.from_coe(
            ATTRACTOR_DIC[orbit.body.name],
            orbit.semi_major_axis * u.m,
            orbit.eccentricity * u.one,
            orbit.inclination * u.rad,
            orbit.longitude_of_ascending_node * u.rad,
            orbit.argument_of_periapsis * u.rad,
            nu * u.rad,
            ut * u.s,
        )

    def launch_window(self, 
                      site_p: np.ndarray, 
                      direction: str = 'SE', 
                      cloest: bool = False, 
                      min_phase: u.Quantity = 10 * u.deg, 
                      max_phase: u.Quantity = 30 * u.deg,
                      start_time: u.Quantity = None, 
                      end_time: u.Quantity = None) -> u.Quantity:
        """发射场向轨道发射的时刻

        Args:
            site_p (ndarray): 发射场位置矢量, 惯性系
            direction (str): 发射方向. Defaults to 'SE'.
            cloest (bool): 返回最近的窗口. Defaults to False.
            min_phase (int, optional): 最小相位差. Defaults to 10.
            max_phase (int, optional): 最大相位差. Defaults to 30.
            start_time (Quantity, optional): 搜索开始时间. Defaults to None.
            end_time (Quantity, optional): 搜索终止时间. Defaults to None.

        Returns:
            float: 发射窗口
        """
        if start_time is None or end_time is None:
            start_time = self.epoch
            end_time = self.epoch + KSP_Earth.rotational_period * 30
        ut = orbit_launch_window(
            self, 
            site_p, 
            direction, 
            cloest, 
            min_phase, 
            max_phase, 
            start_time, 
            end_time
            )
        return ut

    def cheat(self, ut=None):
        if ut is None:
            ut = get_ut() * u.s
        orb = self.propagate_to_epoch(ut)
        nu = orb.nu.to_value(u.rad)
        e = orb.e.to_value(u.one)
        E = nu2E(nu, e)
        Me = E2Me(E, e)
        return (f'SMA: {orb.a.to_value(u.m)}\n'
                f'ECC: {orb.e.to_value(u.one)}\n'
                f'INC: {orb.inc.to_value(u.deg)}\n'
                f'MNA: {Me}\n'
                f'LAN: {orb.raan.to_value(u.deg)}\n'
                f'ARG: {orb.argp.to_value(u.deg)}')
        