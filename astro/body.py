import numpy as np
from functools import cached_property
import threading
from astropy import units as u

from .utils import UTIL_CONN


__all__ = [
    'Body',
    'KSP_Sun',
    'KSP_Earth',
    'KSP_Moon',
    'BODY_DIC',
]


class Body:
    def __init__(self,
                 name: str,
                 attractor,
                 r: u.Quantity,
                 mu: u.Quantity,
                 soi: u.Quantity,
                 rotational_period: u.Quantity,
                 atomsphere_height: u.Quantity = 0 * u.km,
                 ):
        self.name = name
        self.attractor: Body = attractor
        self.r = r
        self.mu = mu
        self.soi = soi
        self.rotational_period = rotational_period
        self.atomsphere_height = atomsphere_height
        self.has_atmosphere = self.atomsphere_height > 0 * u.km

    @property
    def krpc_body(self):
        return UTIL_CONN.space_center.bodies[self.name]
        
    @cached_property
    def angular_velocity(self) -> u.Quantity:
        ref = self.krpc_body.non_rotating_reference_frame
        angular_vel = -np.array(self.krpc_body.angular_velocity(ref), dtype=np.float64) * u.rad / u.s
        angular_vel[1], angular_vel[2] = angular_vel[2], angular_vel[1]
        return angular_vel

    @property
    def orbit(self):
        from .orbit import Orbit
        return Orbit.from_krpcorb(self.krpc_body.orbit)


KSP_Sun = Body(
    name='Sun',
    attractor=None,
    r=None,
    mu=None,
    soi=None,
    rotational_period=None,
)

KSP_Earth = Body(
    name='Earth',
    attractor=KSP_Sun,
    r=6.371e3 * u.km,
    mu=3.9860044615e5 * u.km ** 3 / u.s ** 2,
    soi=924649.216 * u.km,
    rotational_period=86164.1015625 * u.s,
    atomsphere_height = 140 * u.km,
)

KSP_Moon = Body(
    name='Moon',
    attractor=KSP_Earth,
    r=1.7371e3 * u.km,
    mu=4.9028000645e3 * u.km ** 3 / u.s ** 2,
    soi=66167.16 * u.km,
    rotational_period=2370996.25 * u.s,
    atomsphere_height = 0 * u.km,
)


BODY_DIC = {
    'Sun': KSP_Sun,
    'Earth': KSP_Earth,
    'Moon': KSP_Moon,
}
