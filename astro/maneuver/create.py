from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import warnings
from astropy import units as u
from krpc.services.spacecenter import Vessel, Node

from .lambert_solver import *
from .moon_transfer import *
from ..orbit import Orbit
from ..frame import OrbitalFrame, BCI

if TYPE_CHECKING:
    from ..body import *


class Maneuver:
    def __init__(self, impulses, orbit: Orbit):
        """一系列轨道机动

        Args:
            impulses (list[tuple]): 脉冲机动, [(dt, ndarray[v0, v1, v2]), ...]
            orbit (Orbit): 初始轨道
        """
        self._impulses = impulses
        self._orbit = orbit

    @property
    def orbit(self):
        return self._orbit

    def change_orbit(self, orbit: Orbit):
        dt = orbit.epoch - self._orbit.epoch
        for i in range(len(self._impulses)):
            self._impulses[i] = (self._impulses[i][0] - dt, self._impulses[i][1])
        self._orbit = orbit

    def __getitem__(self, key):
        return self._impulses[key]

    def __str__(self):
        s = ''
        for i in range(len(self._impulses)):
            s += f'Impulse {i}: {self._impulses[i][0]}, {self._impulses[i][1]}\n'
        s += f'Total Δv: {self.get_total_cost()}'
        return s

    def get_total_cost(self):
        total = 0
        for i in self._impulses:
            total += np.linalg.norm(i[1])
        return total

    def is_safe(self):
        orbits = self.apply(intermediate=True)
        for i in range(len(orbits) - 1):
            orb = orbits[i]
            safe = orb.is_safe_before(orb.propagate_to_epoch(orbits[i + 1].epoch).nu)
            if not safe:
                return False
        return True

    def apply(self, intermediate=False):
        orbits: list[Orbit] = []
        orb = self.orbit
        epoch = self.orbit.epoch
        for i in self._impulses:
            ut = epoch + i[0]
            orb = orb.propagate_to_epoch(ut)
            v_vec = orb.v_vec + i[1]
            orb = Orbit.from_rv(orb.attractor, orb.r_vec, v_vec, ut)
            orbits.append(orb)
        if intermediate:
            return tuple(orbits)
        return orbits[-1]

    def to_krpcv(self, vessel: Vessel, clear=True) -> list[Node]:
        """创建机动节点

        Args:
            vessel (Vessel): 载具
            clear (boolean): 清除已有的节点

        Returns:
            list[Node]: 创建的节点对象列表
        """
        if clear:
            vessel.control.remove_nodes()
        orbit = self.orbit
        epoch = orbit.epoch
        node_list: list[Node] = []
        for i in self._impulses:
            node_ut = epoch + i[0]
            orbit = orbit.propagate_to_epoch(node_ut)
            orbit_ref = OrbitalFrame.from_orbit(orbit)
            node_burn = orbit_ref.transform_d_from_parent(i[1]).to_value(u.m / u.s)
            krpc_node = vessel.control.add_node(node_ut.to_value(u.s), node_burn[0], node_burn[2], -node_burn[1])
            node_list.append(krpc_node)
            orbit = Orbit.from_krpcorb(krpc_node.orbit, node_ut)
        return node_list
            
    @staticmethod
    def from_krpcv(vessel: Vessel):
        orb = Orbit.from_krpcv(vessel)
        epoch = orb.epoch
        nodes = vessel.control.nodes
        impulses = []
        for n in nodes:
            burn_vector = BCI.transform_d_from_left_hand(n.burn_vector(vessel.orbit.body.non_rotating_reference_frame))
            impulses.append(((n.ut - epoch) * u.s, np.array(burn_vector) * u.m / u.s))
        return Maneuver(impulses, orb)
    
    @staticmethod
    def lambert(orbit_v: Orbit, orbit_t: Orbit, solver=bond, **kwargs):
        k = orbit_v.k.to_value(u.km ** 3 / u.s ** 2)
        r1 = orbit_v.r_vec.to_value(u.km)
        r2 = orbit_t.r_vec.to_value(u.km)
        dt = (orbit_t.epoch - orbit_v.epoch).to_value(u.s)
        try:
            v1, v2 = solver(k, r1, r2, dt, **kwargs)
        except ValueError as e:
            warnings.warn(f"solver '{solver.__name__}' failed: {e}, retrying with other solver", RuntimeWarning, 2)
            v1, v2 = izzo(k, r1, r2, dt, **kwargs)
        imp1 = v1 * u.km / u.s - orbit_v.v_vec
        imp2 = orbit_t.v_vec - v2 * u.km / u.s
        imps = [(0 * u.s, imp1), (dt * u.s, imp2)]
        return Maneuver(imps, orbit_v)

    @staticmethod
    def opt_bi_impulse_rdv(orbit_v: Orbit, orbit_t: Orbit, **kwargs) -> Maneuver | None:
        """双脉冲交会机动能量最优解

        Args:
            orbit_v (Orbit): 初始轨道
            orbit_t (Orbit): 目标初始轨道

        Returns:
            Maneuver: 双脉冲转移机动
        """
        return opt_lambert_by_grid_search(orbit_v, orbit_t, **kwargs)
    
    @staticmethod
    def opt_lambert_multi_revolution(orbit_v: Orbit, orbit_t: Orbit) -> Maneuver | None:
        """lambert问题最佳多周转解

        Args:
            orbit_v (Orbit): 初始轨道
            orbit_t (Orbit): 目标轨道

        Returns:
            Maneuver: 双脉冲转移机动
        """
        return opt_lambert_revolution(orbit_v, orbit_t)

    @classmethod
    def moon_transfer_target(cls, orb_v: Orbit, moon: Body, cap_t: u.Quantity, pe: u.Quantity = 100. * u.km):
        rp_m = pe + moon.r
        orb_target = transfer_target(orb_v, moon, cap_t, orb_v.rp, rp_m, 0 * u.rad)
        return orb_target

    @classmethod
    def moon_transfer(cls,
                      orb_v: Orbit, 
                      moon: Body, 
                      cap_t: u.Quantity,
                      pe: u.Quantity = 100. * u.km):
        """卫星转移机动: 

        Args:
            orb_v (Orbit): 航天器轨道
            target (Body): 卫星
            cap_t (Quantity): 捕获时刻
            pe (Quantity): 捕获轨道的近星点高度

        Returns:
            Maneuver, orbit: 转移机动, 瞄准轨道
        """
        # TODO: 非共面转移
        orb_target = cls.moon_transfer_target(orb_v, moon, cap_t, pe)
        orb_transfer = transfer_start(moon, orb_target, orb_v)
        orbit_start = orb_v.propagate_to_epoch(orb_transfer.epoch)
        t = orb_transfer.epoch - orb_v.epoch
        imp = orb_transfer.v_vec - orbit_start.v_vec
        mnv = Maneuver([(t, imp)], orb_v)
        return mnv, orb_target

    @classmethod
    def course_correction(cls, orb_v: Orbit, orb_t: Orbit, dt=5*u.min):
        orb_mnv_start = orb_v
        orb_mnv_end = orb_t.propagate(-dt)
        mnv = Maneuver.lambert(orb_mnv_start, orb_mnv_end)
        return mnv