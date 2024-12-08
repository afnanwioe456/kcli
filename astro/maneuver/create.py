from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from astropy import units as u
from krpc.services.spacecenter import Vessel, Node

from .lambert_solver import *
from ..orbit import Orbit
from ..frame import OrbitalFrame, ECI

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
        self.orbit = orbit

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

    def apply(self, intermediate=False):
        orbits: list[Orbit] = []
        orb = self.orbit
        epoch = self.orbit.epoch
        for i in self._impulses:
            dt = i[0]
            orb = orb.propagate(dt)
            v_vec = orb.v_vec + i[1]
            orb = Orbit.from_rv(orb.attractor, orb.r_vec, v_vec, epoch + dt)
            orbits.append(orb)
        if intermediate:
            return tuple(orbits)
        return orbits[-1]

    def to_krpcv(self, vessel: Vessel) -> list[Node]:
        """创建机动节点

        Args:
            vessel (Vessel): 载具

        Returns:
            list[Node]: 创建的节点对象列表
        """
        vessel.control.remove_nodes()
        orbit = self.orbit
        epoch = orbit.epoch
        node_list: list[Node] = []
        for i in self._impulses:
            dt = i[0]
            orbit = orbit.propagate(dt)
            orbit_ref = OrbitalFrame.from_orbit(orbit)
            node_burn = orbit_ref.transform_d_from_parent(i[1]).to_value(u.m / u.s)
            node_ut = epoch + dt
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
            burn_vector = ECI.from_left_hand(n.burn_vector(vessel.orbit.body.non_rotating_reference_frame))
            impulses.append(((n.ut - epoch) * u.s, np.array(burn_vector) * u.m / u.s))
        return Maneuver(impulses, orb)
    
    @staticmethod
    def lambert(orbit_1: Orbit, orbit_2: Orbit, solver=bond, **kwargs):
        k = orbit_1.k.to_value(u.km ** 3 / u.s ** 2)
        r1 = orbit_1.r_vec.to_value(u.km)
        r2 = orbit_2.r_vec.to_value(u.km)
        dt = (orbit_2.epoch - orbit_1.epoch).to_value(u.s)
        v1, v2 = solver(k, r1, r2, dt, **kwargs)
        imp1 = v1 * u.km / u.s - orbit_1.v_vec
        imp2 = orbit_2.v_vec - v2 * u.km / u.s
        imps = [(0 * u.s, imp1), (dt * u.s, imp2)]
        return Maneuver(imps, orbit_1)

    @staticmethod
    def opt_bi_impulse_rdv(orbit_v: Orbit, orbit_t: Orbit, **kwargs):
        """双脉冲交会机动能量最优解

        Args:
            orbit_v (Orbit): 初始轨道
            orbit_t (Orbit): 目标初始轨道

        Returns:
            Maneuver: 双脉冲转移机动
        """
        wait_t, trans_t, mnv = opt_lambert_by_grid_search(orbit_v, orbit_t, **kwargs)
        impulses = []
        for i in mnv:
            dt = i[0] + wait_t * u.s
            impulses.append([dt, i[1]])
        return Maneuver(impulses, orbit_v)

    @staticmethod
    def moon_transfer(orbit: Orbit, 
                      target: Body, 
                      pe: u.Quantity = 100. * u.km,
                      e: u.Quantity = 0 * u.one):
        """卫星转移机动
        与卫星共面的航天器首先进行一次调相机动进入转移窗口,
        之后进行一次转移机动, 途中进行两次轨道修正机动,
        在近卫星点进行捕获机动.

        Args:
            orbit (Orbit): 航天器轨道
            target (Body): 卫星
            pe (Quantity): 捕获轨道的近星点高度
            e (Quantity): 停泊轨道的偏心率
        """
    
    