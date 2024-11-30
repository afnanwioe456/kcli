import numpy as np
from poliastro.maneuver import Maneuver as PoliMnv
from astropy import units as u
from krpc.services.spacecenter import Vessel, Node

from .lambert import *
from ..orbit import Orbit
from ..frame import LocalAttitudeFrame, ECI

class Maneuver:
    def __init__(self, polimnv: PoliMnv, orbit: Orbit):
        self._polimnv = polimnv
        self.orbit = orbit

    def __getitem__(self, key, unit=(u.s, u.m/u.s)):
        impulse: tuple[u.quantity.Quantity, np.ndarray[u.quantity.Quantity]] = self._polimnv.impulses[key]
        return impulse[0].to_value(unit[0]), impulse[1].to_value(unit[1])

    def apply(self, intermediate=False):
        if not intermediate:
            poliorb = self.orbit._poliorbit.apply_maneuver(self._polimnv)
            dt = self._polimnv[-1][0].to_value(u.s)
            return Orbit(poliorb, self.orbit.epoch_sec + dt)
        orbits: list[Orbit] = []
        orb = self.orbit
        epoch = self.orbit.epoch_sec
        for i in self._polimnv:
            epoch += i[0].to_value(u.s)
            orb = orb.propagate(epoch - orb.epoch_sec)
            poliorb = orb._poliorbit.apply_maneuver((i, ))
            orb = Orbit(poliorb, epoch)
            orbits.append(orb)
        return tuple(orbits)

    def to_krpcv(self, vessel: Vessel) -> list[Node]:
        """创建机动节点

        Args:
            vessel (Vessel): 载具

        Returns:
            list[Node]: 创建的节点对象列表
        """
        vessel.control.remove_nodes()
        orbit = self.orbit
        epoch = orbit.epoch_sec
        node_list: list[Node] = []
        for i in self._polimnv:
            dt = i[0].to_value(u.s)
            orbit = orbit.propagate(dt)
            orbit_ref = LocalAttitudeFrame.from_orbit(orbit)
            node_burn = orbit_ref.transform_velocity_from_father_frame(
                np.array([0, 0, 0]),
                i[1].to_value(u.m / u.s))
            krpc_node = vessel.control.add_node(epoch + dt, node_burn[0], node_burn[2], -node_burn[1])
            node_list.append(krpc_node)
            orbit = Orbit.from_krpcorb(krpc_node.orbit)
        return node_list
            
    @staticmethod
    def from_krpcv(vessel: Vessel):
        orb = Orbit.from_krpcv(vessel)
        nodes = vessel.control.nodes
        impulses = []
        for n in nodes:
            burn_vector = ECI.from_left_hand(n.burn_vector(vessel.orbit.body.non_rotating_reference_frame))
            impulses.append([n.time_to * u.s, list(burn_vector) * u.m / u.s])
        return Maneuver(PoliMnv(*impulses), orb)
    
    @staticmethod
    def bi_impulse(orbit_v: Orbit, orbit_t: Orbit):
        """双脉冲转移机动能量最优解

        Args:
            orbit_v (Orbit): 初始轨道
            orbit_t (Orbit): 目标轨道

        Returns:
            Maneuver: 双脉冲转移机动
        """
        wait_t, trans_t, mnv = opt_lambert_by_grid_search(orbit_v._poliorbit, orbit_t._poliorbit)
        impulses = []
        for i in mnv:
            dt = i[0] + wait_t * u.s
            impulses.append([dt, i[1]])
        return Maneuver(PoliMnv(*impulses), orbit_v)

    @staticmethod
    def coplanar_moon_transfer(orbit_v: Orbit, orbit_moon: Orbit, alt: float = 100000.):
        """共面卫星转移机动
        与卫星共面的航天器首先进行一次调相机动进入转移窗口,
        之后进行一次转移机动, 在近卫星点进行捕获机动.
        寻找的策略是, 将远/近星点移动至卫星轨道高度,
        反向传播卫星位置, 寻找最佳捕获入射角.

        Args:
            orbit_v (Orbit): 航天器轨道
            orbit_moon (Orbit): 卫星轨道
        """
    
    