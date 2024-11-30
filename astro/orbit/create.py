from functools import cached_property
import numpy as np
from poliastro.twobody import Orbit as PoliOrbit
from poliastro.bodies import Earth
from astropy import time
from astropy import units as u
from krpc.services.spacecenter import Vessel, Orbit as KRPCOrbit

from .utils import orbit_launch_window
from ..utils import get_ut, sec_to_date


ATTRACTOR_DIC = {
    'Earth': Earth,
}

class Orbit:
    def __init__(self,
                 poliorbit: PoliOrbit,
                 epoch_sec: float,
                 ):
        self._poliorbit = poliorbit
        self.epoch_sec = epoch_sec

    @cached_property
    def r(self, unit=u.m):
        r: u.quantity.Quantity = self._poliorbit.r
        return r.to_value(unit)
    
    @cached_property
    def v(self, unit=(u.m/u.s)):
        v: u.quantity.Quantity = self._poliorbit.v
        return v.to_value(unit)
    
    @cached_property
    def a(self, unit=u.m):
        a: u.quantity.Quantity = self._poliorbit.a
        return a.to_value(unit)

    @cached_property
    def ecc(self, unit=u.one):
        ecc: u.quantity.Quantity = self._poliorbit.ecc
        return ecc.to_value(unit)

    @cached_property
    def inc(self, unit=u.rad):
        inc: u.quantity.Quantity = self._poliorbit.inc
        return inc.to_value(unit)

    @cached_property
    def raan(self, unit=u.rad):
        raan: u.quantity.Quantity = self._poliorbit.raan
        return raan.to_value(unit)

    @cached_property
    def argp(self, unit=u.rad):
        argp: u.quantity.Quantity = self._poliorbit.argp
        return argp.to_value(unit)

    @cached_property
    def nu(self, unit=u.rad):
        nu: u.quantity.Quantity = self._poliorbit.nu
        return nu.to_value(unit)

    @cached_property
    def period(self, unit=u.rad):
        period: u.quantity.Quantity = self._poliorbit.period
        return period.to_value(unit)

    def propagate(self, t, unit=u.s):
        quant = t * unit
        epoch_sec = self.epoch_sec + quant.to_value(u.s)
        orb = self._poliorbit.propagate(quant)
        return Orbit(orb, epoch_sec)

    @classmethod
    def from_coe(cls, attractor, a, ecc, inc, raan, argp, nu, epoch_sec):
        """从经典轨道元素和KSPRO历时创建Orbit对象

        Args:
            attractor str: 中心天体
            a float: 半长轴
            ecc float: 离心率
            inc float: 轨道倾角
            raan float: 升交点经度
            argp float: 近地点辐角
            nu float: 真近点角
            epoch_sec int: KSPRO历元时刻, 自1951-01-01 00:00:00以来的秒数
        """
        o = PoliOrbit.from_classical(ATTRACTOR_DIC[attractor],
                                     a * u.m,
                                     ecc * u.one,
                                     inc * u.rad,
                                     raan * u.rad,
                                     argp * u.rad,
                                     nu * u.rad,
                                     time.Time(sec_to_date(epoch_sec)))
        return cls(o, epoch_sec)
    
    @classmethod
    def from_krpcv(cls, vessel: Vessel):
        """从krpc Vessel对象创建Orbit对象

        Args:
            vessel (Vessel): krpc Vessel
        """
        orbit = vessel.orbit
        ut = get_ut()
        nu = orbit.true_anomaly
        return cls.from_coe(orbit.body.name,
                            orbit.semi_major_axis,
                            orbit.eccentricity,
                            orbit.inclination,
                            orbit.longitude_of_ascending_node,
                            orbit.argument_of_periapsis,
                            nu,
                            ut)


    @classmethod
    def from_krpcorb(cls, orbit: KRPCOrbit):
        """从krpc Orbit对象创建Orbit对象

        Args:
            orbit (Orbit): krpc Orbit
        """
        ut = get_ut()
        return cls.from_coe(orbit.body.name,
                            orbit.semi_major_axis,
                            orbit.eccentricity,
                            orbit.inclination,
                            orbit.longitude_of_ascending_node,
                            orbit.argument_of_periapsis,
                            orbit.true_anomaly_at_ut(ut),
                            ut)

    def launch_window(self, site_p, direction='SE', phase_diff=40, start_period=0, end_period=30):
        """返回发射场向轨道发射的最佳ut

        Args:
            site_p (ndarray): 发射场位置矢量, 惯性系
            direction (str): 发射方向SE/NE/[Default]. Defaults to 'SE'.
            phase_diff (int, optional): 最小相位差. Defaults to 40.
            start_period (int, optional): 开始周期. Defaults to 0.
            end_period (int, optional): 结束周期. Defaults to 30.

        Returns:
            float: 发射窗口
        """
        ut: float = orbit_launch_window(self, np.array(site_p), direction, phase_diff, start_period, end_period)
        return ut

    
        


