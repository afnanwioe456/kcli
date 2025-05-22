from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from astropy import units as u
from krpc.services.spacecenter import Vessel, Node

from .lambert_solver import *
from .moon_transfer import *
from .planner import *
from ..orbit import Orbit
from ..frame import OrbitalFrame, BCIFrame

if TYPE_CHECKING:
    from ..body import *


class Maneuver:
    def __init__(self, impulses, orbit: Orbit):
        """轨道机动序列

        Args:
            impulses (list[tuple]): 脉冲机动, [(dt, ndarray[v0, v1, v2]), ...]
            orbit (Orbit): 初始轨道
        """
        self._impulses: list[tuple[u.Quantity]] = impulses
        self._orbit = orbit

    def __getitem__(self, key):
        return self._impulses[key]

    def __str__(self):
        s = ''
        for i in range(len(self._impulses)):
            s += (f'Impulse {i}: {round(self._impulses[i][0].to_value(u.s), 2)} s, '
                  f'{np.round(self._impulses[i][1].to_value(u.km / u.s), 3)} km/s\n')
        s += f'Total Δv: {round(self.get_total_cost().to_value(u.km / u.s), 3)} km/s'
        return s

    @property
    def orbit(self):
        return self._orbit

    def change_orbit(self, orbit: Orbit):
        dt = orbit.epoch - self._orbit.epoch
        imps = []
        for i in range(len(self._impulses)):
            imps.append((self._impulses[i][0] - dt, self._impulses[i][1]))
        return Maneuver(imps, orbit)
        
    def merge(self, mnv: Maneuver):
        if len(mnv._impulses) == 0:
            return self
        mnv = mnv.change_orbit(self.orbit)
        if len(self._impulses) == 0:
            return mnv
        if abs(self._impulses[-1][0] - mnv._impulses[0][0]) < 1 * u.s:
            # 如果两个脉冲几乎重叠
            self._impulses[-1] = (self._impulses[-1][0],
                                  self._impulses[-1][1] + mnv._impulses[0][1])
            self._impulses += mnv._impulses[1:]
        else:
            self._impulses += mnv._impulses
        return self

    @staticmethod
    def serial(orbit: Orbit, mnvs: list[Maneuver]):
        res = mnvs[0].change_orbit(orbit)
        for i in range(1, len(mnvs)):
            mnv = mnvs[i].change_orbit(orbit)
            res = res.merge(mnv)
        return res

    def get_total_cost(self) -> u.Quantity:
        total = 0
        for i in self._impulses:
            total += np.linalg.norm(i[1])
        return total

    def is_safe(self):
        orbits = self.apply(intermediate=True)
        for i in range(len(orbits) - 1):
            orb = orbits[i]
            safe = orb.is_safe_before(orbits[i + 1].epoch)
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
        return orbits[-1] if len(orbits) > 0 else self.orbit

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
            burn_vector = BCIFrame.transform_d_from_left_hand(n.burn_vector(vessel.orbit.body.non_rotating_reference_frame))
            impulses.append(((n.ut - epoch) * u.s, np.array(burn_vector) * u.m / u.s))
        return Maneuver(impulses, orb)

    @staticmethod
    def change_apoapsis(orbit: Orbit, ap: u.Quantity, immediate: bool = False):
        imps = apoapsis_planner(orbit, ap, immediate)
        return Maneuver(imps, orbit)

    @staticmethod
    def change_periapsis(orbit: Orbit, pe: u.Quantity, immediate: bool = False):
        imps = periapsis_planner(orbit, pe, immediate)
        return Maneuver(imps, orbit)

    @staticmethod
    def match_plane(orb_v: Orbit, 
                    orb_t: Orbit, 
                    closest: bool = False,
                    conserved: bool = True):
        imps = match_plane_planner(orb_v, orb_t, closest=closest, conserved=conserved)
        return Maneuver(imps, orb_v)

    @staticmethod
    def change_phase(orb: Orbit,
                     revisit_epoch: u.Quantity,
                     at_pe = False,
                     safety_check = True,
                     conserved = True):
        """调相机动

        Args:
            orb (Orbit): 航天器轨道
            revisit_epoch (u.Quantity): 重访时刻
            at_pe (bool, optional): 允许近星点处的转移. Defaults to False.
            safety_check (bool, optional): 安全检测. Defaults to True.
            conserved (bool, optional): 恢复原轨道. Defaults to True.

        Raises:
            ValueError: 无解, 尝试兰伯特机动

        Returns:
            Maneuver: 调相机动
        """
        dt = revisit_epoch - orb.epoch
        M = dt // orb.period
        threshold = 20
        if M >= threshold:  # 周期数过多可能导致机动过小
            revisit_epoch = orb.epoch + threshold * orb.period + dt % orb.period
            conserved = True
        mnvs: list[Maneuver] = []
        inner_imps = change_phase_planner(orb, revisit_epoch, True, conserved=conserved)
        if inner_imps is not None:
            mnvs.append(Maneuver(inner_imps, orb))
        outer_imps = change_phase_planner(orb, revisit_epoch, False, conserved=conserved)
        if outer_imps is not None:
            mnvs.append(Maneuver(outer_imps, orb))
        if at_pe:
            orb_pe = orb.propagate_to_nu(0 * u.rad, prograde=True)
            dt = orb_pe.epoch - orb.epoch
            revisit_pe = revisit_epoch - (orb.period - dt)
            inner_imps = change_phase_planner(orb_pe, revisit_pe, True, conserved=True)
            if inner_imps is not None:
                mnvs.append(Maneuver(inner_imps, orb_pe).change_orbit(orb))
            outer_imps = change_phase_planner(orb_pe, revisit_pe, False, conserved=True)
            if outer_imps is not None:
                mnvs.append(Maneuver(outer_imps, orb_pe).change_orbit(orb))
        min_dv = np.inf * u.km / u.s
        best_mnv = None
        for m in mnvs:
            if safety_check and not m.is_safe():
                continue
            dv = m.get_total_cost()
            if dv < min_dv:
                min_dv = dv
                best_mnv = m
        if best_mnv is None:
            raise ValueError('No possible solution')
        return best_mnv
    
    @staticmethod
    def lambert(orb_v: Orbit, orb_t: Orbit, solver=bond, **kwargs):
        imps = lambert_planner(orb_v, orb_t, solver=solver, **kwargs)
        return Maneuver(imps, orb_v)

    @staticmethod
    def opt_bi_impulse_rdv(orbit_v: Orbit, 
                           orbit_t: Orbit, 
                           safety_check: bool = True,
                           before: u.Quantity | None = None,
                           ) -> Maneuver | None:
        """双脉冲交会机动能量最优解

        Args:
            orbit_v (Orbit): 初始轨道
            orbit_t (Orbit): 目标初始轨道

        Returns:
            Maneuver: 双脉冲转移机动
        """
        return opt_lambert_by_sa(orbit_v, 
                                 orbit_t, 
                                 safety_check=safety_check, 
                                 before=before)
    
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
    def course_correction(cls, orb_v: Orbit, orb_t: Orbit, dt=5*u.min):
        orb_mnv_start = orb_v
        orb_mnv_end = orb_t.propagate(-dt)
        mnv = Maneuver.lambert(orb_mnv_start, orb_mnv_end)
        return mnv
        
    @classmethod
    def moon_transfer_target(cls, 
                             orb_v: Orbit, 
                             moon: Body, 
                             cap_t: u.Quantity, 
                             pe: u.Quantity = 100 * u.km,
                             inc: u.Quantity = 0 * u.rad,
                             relative: bool = True):
        """卫星转移瞄准轨道

        Args:
            orb_v (Orbit): 航天器轨道
            moon (Body): 卫星
            cap_t (u.Quantity): 捕获时刻
            pe (u.Quantity, optional): 捕获轨道的近星点高度. Defaults to 100*u.km.
            inc (u.Quantity, optional): 捕获轨道倾角. Defaults to 0*u.rad.
            relative (bool, optional): 使用相对倾角. Defaults to True.

        Returns:
            Orbit: 转移瞄准轨道
        """
        raise NotImplementedError()  # 废弃
        orb_target = transfer_target(orb_v, moon, cap_t, pe, inc, relative=relative)
        orb_transfer = orb_target.propagate_to_nu(0*u.rad, True, -1)
        if orb_transfer.epoch < orb_v.epoch:
            cap_t += orb_v.epoch - orb_transfer.epoch + 600 * u.s
            warnings.warn(f'orbit already passed transfer window, trying capture time {cap_t}', RuntimeWarning)
            return cls.moon_transfer_target(orb_v, moon, cap_t, pe=pe, inc=inc, relative=relative)
        return orb_target

    @classmethod
    def moon_orbit_transfer_target(cls, 
                                   orb_v: Orbit, 
                                   orb_t: Orbit, 
                                   cap_t: u.Quantity,
                                   rp_t: u.Quantity | None = None
                                   ):
        """向卫星目标轨道转移的瞄准轨道, 位于捕获临界前

        Args:
            orb_v (Orbit): 航天器轨道
            orb_t (Orbit): 目标轨道
            cap_t (Quantity): 瞄准捕获时间
            rp_m (Quantity): 飞越近星点

        Returns:
            Orbit: 瞄准轨道
        """
        # 瞄准捕获时间当前只是开始搜索时间, 因此如果瞄准时间大于最近的窗口, 可能会错过较近的窗口
        orb_target = transfer_orbit_target(orb_v, orb_t, cap_t, rp_t=rp_t)
        orb_transfer = orb_target.propagate_to_nu(0*u.rad, prograde=True, M=-1)
        if orb_transfer.epoch < orb_v.epoch:
            cap_t += orb_t.attractor.rotational_period / 2
            warnings.warn(f'orbit already passed transfer window, trying capture time {cap_t}', RuntimeWarning)
            return cls.moon_orbit_transfer_target(orb_v, orb_t, cap_t)
        return orb_target

    @classmethod
    def moon_return_target(cls, orb_v: Orbit, pe: u.Quantity, esc_t: u.Quantity):
        """从卫星返回行星指定近星点高度的瞄准轨道, 位于逃逸临界前

        Args:
            orb_v (Orbit): 航天器轨道
            pe (Quantity): 近星点高度
            esc_t (Quantity): 瞄准逃逸时间

        Returns:
            Orbit: 瞄准轨道
        """
        orb_target = return_target(orb_v, pe, esc_t)
        orb_transfer = orb_target.propagate_to_nu(0 * u.rad, prograde=True, M=-1)
        if orb_transfer.epoch < orb_v.epoch:
            esc_t += orb_v.epoch - orb_transfer.epoch + 600 * u.s
            warnings.warn(f'orbit already passed transfer window, trying capture time {esc_t}', RuntimeWarning)
            return cls.moon_return_target(orb_v, pe, esc_t)

    @classmethod
    def transfer(cls, orb_v: Orbit, orb_t: Orbit):
        """轨道转移机动: 

        Args:
            orb_v (Orbit): 航天器轨道
            orb_t (Orbit): 瞄准轨道

        Returns:
            Maneuver: 转移机动
        """
        # 先匹配轨道倾角, 再进行一次调相机动进入转移窗口
        orb_transfer = orb_t.propagate_to_nu(0*u.rad, prograde=True, M=-1)
        mnv_coplanar = Maneuver.match_plane(orb_v, orb_transfer, closest=True, conserved=True)
        orb_coplanar = mnv_coplanar.apply()
        transfer_nu = orb_coplanar.nu_at_direction(orb_transfer.r_vec)
        orb_coplanar = orb_coplanar.propagate_to_nu(transfer_nu, True)
        mnv_phase = Maneuver.change_phase(orb_coplanar, orb_transfer.epoch, at_pe=True, safety_check=True, conserved=False)
        orb_start = mnv_phase.apply()
        orb_start = orb_start.propagate_to_epoch(orb_transfer.epoch)
        mnv_transfer = Maneuver.lambert(orb_start, orb_t.propagate(-5 * u.min))
        mnv_series = mnv_coplanar.merge(mnv_phase).merge(mnv_transfer)
        return mnv_series
