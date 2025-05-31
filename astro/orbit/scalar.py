from __future__ import annotations
from typing import TYPE_CHECKING
from functools import cached_property
import numpy as np

from ..core.kepler import *
from ..utils import sec_to_date
from ...math import vector as mv

if TYPE_CHECKING:
    from ..body import Body

class OrbitBase:
    def __init__(self):
        self.attractor: Body = None
        self._epoch: float = None
        self._a: float = None
        self._h: float = None
        self._e: float = None
        self._inc: float = None
        self._raan: float = None
        self._argp: float = None
        self._nu: float = None
        self._r_vec: np.ndarray = None
        self._v_vec: np.ndarray = None
    
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
            h = a2h(self.a,
                    self.attractor.mu,
                    self.e)
            self._h = h
        return self._h

    @cached_property
    def h_vec(self):
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
    def r_vec(self):
        if self._r_vec is None:
            self._to_rv()
        return self._r_vec

    @cached_property
    def r(self):
        return np.linalg.norm(self.r_vec)

    @property
    def v_vec(self):
        if self._v_vec is None:
            self._to_rv()
        return self._v_vec

    @cached_property
    def v(self):
        return np.linalg.norm(self.v_vec)

    @cached_property
    def vt(self):
        return np.linalg.norm(self.vt_vec)

    @cached_property
    def vt_vec(self):
        return self.v_vec - self.vr_vec

    @cached_property
    def vr(self):
        return np.linalg.norm(self.vr_vec)

    @cached_property
    def vr_vec(self):
        r_i = mv.normalize(self.r_vec)
        return np.dot(self.v_vec, r_i) * r_i

    @cached_property
    def period(self) -> float:
        if self.e >= 1:
            return np.inf
        return a2T(self.a, self.attractor.mu)

    @cached_property
    def delta_t(self):
        """从近地点到真近点角的Δt"""
        return self.delta_t_at_nu(self.nu)
    
    def delta_t_at_nu(self, nu) -> float:
        """从近地点到真近点角的Δt"""
        if self.e < 1:
            return nu2dt_e(nu, self.e, self.period)
        return nu2dt_h(nu, self.e, self.attractor.mu, self.h)
    
    def r_at_nu(self, nu) -> float:
        nu = nu % (2 * np.pi)
        return nu2r(nu, self.h, self.e, self.attractor.mu)

    @cached_property
    def rp(self):
        return self.r_at_nu(0)

    @cached_property
    def ra(self):
        return self.r_at_nu(np.pi)

    @cached_property
    def ap(self):
        return self.ra - self.attractor.r
    
    @cached_property
    def pe(self):
        return self.rp - self.attractor.r

    def is_safe(self, nu=0) -> bool:
        nu = nu % (2 * np.pi)
        rp = self.r_at_nu(nu)
        limit = self.attractor.r + self.attractor.atmosphere_height
        return rp > limit

    def _to_rv(self):
        if self._has_rv_state():
            pass
        elif self._has_coe_state():
            r_vec, v_vec = coe2rv(
                self.h,
                self.e,
                self.inc,
                self.raan,
                self.argp,
                self.nu,
                self.attractor.mu
            )
            self._r_vec = r_vec
            self._v_vec = v_vec
        else:
            raise NotImplementedError()
        return self
    
    def _to_coe(self):
        if self._has_coe_state():
            pass
        elif self._has_rv_state():
            self._h, self._e, self._inc, self._raan, self._argp, self._nu = \
                rv2coe(self._r_vec, self._v_vec, self.attractor.mu)
            a = h2a(self.h, self.attractor.mu, self.e)
            self._a = a
        else: 
            raise NotImplementedError()
        return self

    def _has_rv_state(self):
        return self._r_vec is not None and self._v_vec is not None

    def _has_coe_state(self):
        return self._a is not None and self._e is not None and self._inc is not None \
            and self._raan is not None and self._argp is not None and self._nu is not None

    def _assign_coe(self, attractor, a, e, inc, raan, argp, nu, epoch):
        inc     = inc % (2 * np.pi)
        if inc > np.pi:
            inc     = 2 * np.pi - inc
            raan    = (raan + np.pi) % (2 * np.pi)
            argp    = (argp + np.pi) % (2 * np.pi)
        raan    = raan % (2 * np.pi)
        argp    = argp % (2 * np.pi)
        nu      = nu % (2 * np.pi)
        self.attractor  = attractor
        self._epoch     = epoch
        self._a, self._e, self._inc, self._raan, self._argp, self._nu = \
            a, e, inc, raan, argp, nu

    def _assign_rv(self, attarctor, r_vec, v_vec, epoch):
        self.attractor = attarctor
        self._r_vec = r_vec
        self._v_vec = v_vec
        self._epoch = epoch
        
    def __repr__(self):
        return (f"Orbit {self.pe / 1000:.2f} km * "
                f"{self.ap / 1000:.2f} km * "
                f"{np.rad2deg(self.inc):.2f} deg "
                f"around {self.attractor.name} "
                f"at {sec_to_date(int(self.epoch))}")
