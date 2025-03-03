import numpy as np
from functools import cached_property
from astropy import units as u

from .utils import UTIL_CONN


__all__ = [
    'Body',
    'KSP_Earth',
    'KSP_Moon',
    'BODY_DIC',
]


class Body:
    def __init__(self,
                 name: str,
                 attractor,
                 r: u.Quantity,
                 k: u.Quantity,
                 soi: u.Quantity,
                 rotational_period: u.Quantity,
                 atomsphere_height: u.Quantity = 0 * u.km,
                 ):
        self.name = name
        self.attractor = attractor
        self.r = r
        self.k = k
        self.soi = soi
        self.rotational_period = rotational_period
        self.atomsphere_height = atomsphere_height
        
    @cached_property
    def angular_velocity(self):
        return (2 * np.pi * u.rad) / self.rotational_period.to(u.s)

    @property
    def orbit(self):
        from .orbit import Orbit
        body = UTIL_CONN.space_center.bodies[self.name]
        return Orbit.from_krpcorb(body.orbit)


KSP_Earth = Body(
    'Earth',
    None,
    6.371e3 * u.km,
    3.9860044615e5 * u.km ** 3 / u.s ** 2,
    924649.216 * u.km,
    86164.1015625 * u.s,
    atomsphere_height = 140 * u.km,
)

KSP_Moon = Body(
    'Moon',
    KSP_Earth,
    1.7371e3 * u.km,
    4.9028000645e3 * u.km ** 3 / u.s ** 2,
    66167.16 * u.km,
    2370996.25 * u.s,
    atomsphere_height = 0 * u.km,
)


BODY_DIC = {
    'Earth': KSP_Earth,
    'Moon': KSP_Moon,
}
