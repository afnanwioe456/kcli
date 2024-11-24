from functools import cached_property
import krpc
from poliastro.twobody import Orbit as PoliOrbit
from poliastro.bodies import Earth
from astropy import time
from astropy import units as u
from krpc.services.spacecenter import Vessel

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
    def rann(self, unit=u.rad):
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
        return cls.from_coe(orbit.body.name,
                            orbit.semi_major_axis,
                            orbit.eccentricity,
                            orbit.inclination,
                            orbit.longitude_of_ascending_node,
                            orbit.argument_of_periapsis,
                            orbit.true_anomaly,
                            get_ut())


    
        


