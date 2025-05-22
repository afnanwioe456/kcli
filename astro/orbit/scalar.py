from __future__ import annotations
from typing import TYPE_CHECKING
from functools import cached_property
import numpy as np
from astropy import units as u

from ..core.kepler import *
from ..utils import sec_to_date
from ...math import vector as mv

if TYPE_CHECKING:
    from ..body import Body

class OrbitBase:
    def __init__(self):
        self.attractor: Body = None
        self._epoch: u.Quantity = None
        self._a: u.Quantity = None
        self._h: u.Quantity = None
        self._e: u.Quantity = None
        self._inc: u.Quantity = None
        self._raan: u.Quantity = None
        self._argp: u.Quantity = None
        self._nu: u.Quantity = None
        self._r_vec: np.ndarray[u.Quantity] = None
        self._v_vec: np.ndarray[u.Quantity] = None
    
    @property
    def epoch(self):
        return self._epoch

    @property
    def a(self):
        if self._a is None:
            self._to_coe()
        return self._a
            
    @property
    def h(self):
        if self._h is None:
            h = a2h(self.a.to_value(u.km),
                    self.attractor.mu.to_value(u.km ** 3 / u.s ** 2),
                    self.e.to_value(u.one))
            self._h = h * u.km ** 2 / u.s
        return self._h

    @cached_property
    def h_vec(self) -> u.Quantity:
        return np.cross(self.r_vec, self.v_vec)

    @property
    def e(self):
        if self._e is None:
            self._to_coe()
        return self._e

    @property
    def inc(self):
        if self._inc is None:
            self._to_coe()
        return self._inc

    @property
    def raan(self):
        if self._raan is None:
            self._to_coe()
        return self._raan

    @property
    def argp(self):
        if self._argp is None:
            self._to_coe()
        return self._argp

    @property
    def nu(self):
        if self._nu is None:
            self._to_coe()
        return self._nu

    @property
    def coe(self):
        return self.a, self.e, self.inc, self.raan, self.argp, self.nu

    @property
    def r_vec(self) -> u.Quantity:
        if self._r_vec is None:
            self._to_rv()
        return self._r_vec

    @cached_property
    def r(self) -> u.Quantity:
        return np.linalg.norm(self.r_vec)

    @property
    def v_vec(self) -> u.Quantity:
        if self._v_vec is None:
            self._to_rv()
        return self._v_vec

    @cached_property
    def v(self) -> u.Quantity:
        return np.linalg.norm(self.v_vec)

    @cached_property
    def vt(self) -> u.Quantity:
        return np.linalg.norm(self.vt_vec)

    @cached_property
    def vt_vec(self) -> u.Quantity:
        return self.v_vec - self.vr_vec

    @cached_property
    def vr(self) -> u.Quantity:
        return np.linalg.norm(self.vr_vec)

    @cached_property
    def vr_vec(self) -> u.Quantity:
        r_i = mv.normalize(self.r_vec)
        return np.dot(self.v_vec, r_i) * r_i

    @cached_property
    def period(self) -> u.Quantity:
        if self.e >= 1:
            return np.inf * u.s
        return T(
            self.a.to_value(u.km),
            self.attractor.mu.to_value(u.km ** 3 / u.s ** 2)
            ) * u.s

    @cached_property
    def delta_t(self) -> u.Quantity:
        """从近地点到真近点角的Δt"""
        return self.delta_t_at_nu(self.nu)
    
    def delta_t_at_nu(self, nu) -> u.Quantity:
        """从近地点到真近点角的Δt"""
        if self.e < 1:
            return nu2dt_e(
                nu.to_value(u.rad), 
                self.e.to_value(u.one),
                self.period.to_value(u.s)
                ) * u.s
        return nu2dt_h(
            nu.to_value(u.rad), 
            self.e.to_value(u.one), 
            self.attractor.mu.to_value(u.km ** 3 / u.s ** 2), 
            self.h.to_value(u.km ** 2 / u.s)
            ) * u.s
    
    def r_at_nu(self, nu) -> u.Quantity:
        nu = self._correct_coe(nu)
        return nu2r(
            nu.to_value(u.rad),
            self.h.to_value(u.km ** 2 / u.s),
            self.e.to_value(u.one),
            self.attractor.mu.to_value(u.km ** 3 / u.s ** 2)
        ) * u.km

    def nu_at_direction(self, r_vec) -> u.Quantity:
        from ..frame import PQWFrame
        r_vec = r_vec.to_value(u.km)
        r_vec = PQWFrame.from_orbit(self).transform_d_from_parent(r_vec)
        nu = np.arctan2(r_vec[1], r_vec[0])  # arctan象限不清
        if nu < 0:
            nu += 2 * np.pi
        return nu * u.rad

    @cached_property
    def rp(self):
        return self.r_at_nu(0 * u.rad)

    @cached_property
    def ra(self):
        return self.r_at_nu(np.pi * u.rad)

    @cached_property
    def ap(self) -> u.Quantity:
        return self.ra - self.attractor.r
    
    @cached_property
    def pe(self) -> u.Quantity:
        return self.rp - self.attractor.r

    def is_safe(self, nu=0*u.rad) -> bool:
        nu = self._correct_coe(nu)
        rp = self.r_at_nu(nu)
        limit = self.attractor.r + self.attractor.atomsphere_height
        return rp > limit

    def _to_rv(self):
        if self._has_rv_state():
            pass
        elif self._has_coe_state():
            r_vec, v_vec = coe2rv(
                self.h.to_value(u.km ** 2 / u.s),
                self.e.to_value(u.one),
                self.inc.to_value(u.rad),
                self.raan.to_value(u.rad),
                self.argp.to_value(u.rad),
                self.nu.to_value(u.rad),
                self.attractor.mu.to_value(u.km ** 3 / u.s ** 2))
            self._r_vec = r_vec * u.km
            self._v_vec = v_vec * u.km / u.s
        else:
            raise NotImplementedError()
        return self
    
    def _to_coe(self):
        if self._has_coe_state():
            pass
        elif self._has_rv_state():
            h, e, inc, raan, argp, nu = rv2coe(self._r_vec.to_value(u.km),
                                               self._v_vec.to_value(u.km / u.s),
                                               self.attractor.mu.to_value(u.km ** 3/ u.s ** 2))
            self._h = h * u.km ** 2 / u.s
            self._e = e * u.one
            self._inc = inc * u.rad
            self._raan = raan * u.rad
            self._argp = argp * u.rad
            self._nu = nu * u.rad
            a = h2a(self.h.to_value(u.km ** 2 / u.s),
                    self.attractor.mu.to_value(u.km ** 3 / u.s ** 2),
                    self.e.to_value(u.one))
            self._a = a * u.km
        else: 
            raise NotImplementedError()
        return self

    def _has_rv_state(self):
        return self._r_vec is not None and self._v_vec is not None

    def _has_coe_state(self):
        return self._a is not None and self._e is not None and self._inc is not None \
            and self._raan is not None and self._argp is not None and self._nu is not None

    def _correct_coe(self, e):
        e = e % (2 * np.pi * u.rad)
        return e
    
    def _correct_inc(self, e):
        e = e % (np.pi * u.rad)
        return e

    def _assign_coe(self, attractor, a, e, inc, raan, argp, nu, epoch):
        inc = self._correct_inc(inc)
        raan = self._correct_coe(raan)
        argp = self._correct_coe(argp)
        nu = self._correct_coe(nu)
        self.attractor = attractor
        self._epoch = epoch
        self._a, self._e, self._inc, self._raan, self._argp, self._nu = \
            a, e, inc, raan, argp, nu

    def _assign_rv(self, attarctor, r_vec, v_vec, epoch):
        self.attractor = attarctor
        self._epoch = epoch
        self._r_vec, self._v_vec = r_vec, v_vec
        
    def __str__(self):
        return (f"Orbit {round(self.pe.to_value(u.km), 2)} * "
                f"{round(self.ap.to_value(u.km), 2)} km "
                f"around {self.attractor.name} "
                f"at {sec_to_date(self.epoch)}")
