from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from krpc.services.spacecenter import Vessel, Node

from .lambert_solver import *
from .moon_transfer import *
from .planner import *
from ..orbit import Orbit
from ..frame import OrbitalFrame, BCIFrame

if TYPE_CHECKING:
    from ..body import *


class Maneuver:
    def __init__(self, 
                 impulses: list[tuple[float, np.ndarray]], 
                 orbit: Orbit):
        """轨道机动序列

        Args:
            impulses (list[tuple[float, ndarray]]): 脉冲机动
            orbit (Orbit): 初始轨道
        """
        self._impulses = impulses
        self._orbit = orbit

    def __getitem__(self, key):
        return self._impulses[key]

    def __str__(self):
        s = ''
        for i in range(len(self._impulses)):
            s += (f'Impulse {i}: {self._impulses[i][0]:.2f} s, '
                  f'{np.round(self._impulses[i][1], 2)} m/s\n')
        s += f'Total Δv: {round(self.get_total_cost(), 2)} m/s'
        return s

    @property
    def orbit(self):
        return self._orbit

    def change_orbit(self, orbit: Orbit) -> Maneuver:
        dt = orbit.epoch - self._orbit.epoch
        imps = []
        for i in range(len(self._impulses)):
            imps.append((self._impulses[i][0] - dt, self._impulses[i][1]))
        return Maneuver(imps, orbit)
        
    def merge(self, mnv: Maneuver, tol=1) -> Maneuver:
        if len(mnv._impulses) == 0:
            return self
        mnv = mnv.change_orbit(self.orbit)
        if len(self._impulses) == 0:
            return mnv
        imp = self._impulses.copy()
        if abs(imp[-1][0] - mnv._impulses[0][0]) < tol:
            # FIXME: 如果两个脉冲几乎重叠
            imp[-1] = (imp[-1][0], imp[-1][1] + mnv._impulses[0][1])
            imp += mnv._impulses[1:]
        else:
            imp += mnv._impulses
        return Maneuver(imp, self.orbit)

    @staticmethod
    def serial(orbit: Orbit, mnvs: list[Maneuver]) -> Maneuver:
        res = mnvs[0].change_orbit(orbit)
        for i in range(1, len(mnvs)):
            mnv = mnvs[i].change_orbit(orbit)
            res = res.merge(mnv)
        return res

    def get_total_cost(self) -> float:
        total = 0
        for i in self._impulses:
            total += np.linalg.norm(i[1])
        return total

    def is_safe(self, end_orbit=False):
        orbits = self.apply(intermediate=True)
        size = len(orbits) - 1
        for i in range(size):
            orb = orbits[i]
            if not orb.is_safe_before(orbits[i + 1].epoch):
                return False
        if end_orbit and not orbits[-1].is_safe():
            return False
        return True

    def apply(self, intermediate=False) -> list[Orbit] | Orbit:
        orbits      = []
        orb         = self.orbit
        epoch       = self.orbit.epoch
        for i in self._impulses:
            ut          = epoch + i[0]
            orb         = orb.propagate_to_epoch(ut)
            v_vec       = orb.v_vec + i[1]
            orb         = Orbit.from_rv(orb.attractor, orb.r_vec, v_vec, ut)
            orbits.append(orb)
        if intermediate:
            return orbits
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
        orbit       = self.orbit
        epoch       = orbit.epoch
        node_list   = []
        for i in self._impulses:
            node_ut     = epoch + i[0]
            orbit       = orbit.propagate_to_epoch(node_ut)
            orbit_ref   = OrbitalFrame.from_orbit(orbit)
            node_burn   = orbit_ref.transform_d_from_parent(i[1])
            krpc_node   = vessel.control.add_node(node_ut, node_burn[0], node_burn[2], -node_burn[1])
            node_list.append(krpc_node)
            orbit       = Orbit.from_rv(orbit.attractor, orbit.r_vec, orbit.v_vec + i[1], orbit.epoch)
        return node_list
            
    @staticmethod
    def from_krpcv(vessel: Vessel):
        orb         = Orbit.from_krpcv(vessel)
        epoch       = orb.epoch
        nodes       = vessel.control.nodes
        impulses    = []
        for n in nodes:
            burn_vector = n.burn_vector(vessel.orbit.body.non_rotating_reference_frame)
            burn_vector = BCIFrame.transform_d_from_left_hand(burn_vector)
            impulses.append((n.ut - epoch, np.array(burn_vector)))
        return Maneuver(impulses, orb)

    @staticmethod
    def change_apoapsis(orbit: Orbit, ap: float, immediate: bool = False):
        imps = apoapsis_planner(orbit, ap, immediate)
        return Maneuver(imps, orbit)

    @staticmethod
    def change_periapsis(orbit: Orbit, pe: float, immediate: bool = False):
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
                     revisit_epoch: float,
                     immediate = False,
                     conserved = True,
                     safety_check = True) -> Maneuver | None:
        """调相机动

        Args:
            orb (Orbit): 航天器轨道
            revisit_epoch (float): 重访时刻
            at_pe (bool, optional): 允许近星点处的转移. Defaults to False.
            conserved (bool, optional): 恢复原轨道. Defaults to True.
            safety_check (bool, optional): 安全检测. Defaults to True.

        Returns:
            Maneuver: 调相机动
        """
        dt = revisit_epoch - orb.epoch
        period = orb.period
        M = dt // period
        rmd = dt % period
        if rmd < 1e-2:
            return Maneuver([], orb)
        threshold = rmd // 10 + 1
        if M > threshold:  
            # 如果每圈的调整不足10s, 降低周期数
            dt = threshold * period + rmd
            conserved = True

        mnvs: list[Maneuver] = []
        inner_imps = change_phase_planner(orb, dt, True, conserved=conserved)
        if inner_imps is not None:
            mnvs.append(Maneuver(inner_imps, orb))
        outer_imps = change_phase_planner(orb, dt, False, conserved=conserved)
        if outer_imps is not None:
            mnvs.append(Maneuver(outer_imps, orb))
        if not immediate and dt > period:
            # 如果不要求立即调相, 传播到近地点寻找是否有更高效率的脉冲
            orb_new = orb.propagate_to_nu(0)
            dt = dt - period
            inner_imps = change_phase_planner(orb_new, dt, True, conserved=True)
            if inner_imps is not None:
                mnvs.append(Maneuver(inner_imps, orb_new).change_orbit(orb))
            outer_imps = change_phase_planner(orb_new, dt, False, conserved=True)
            if outer_imps is not None:
                mnvs.append(Maneuver(outer_imps, orb_new).change_orbit(orb))

        min_dv = np.inf
        best_mnv = None
        for m in mnvs:
            # FIXME: 是否要对最终轨道进行安全检测?
            if safety_check and not m.is_safe(end_orbit=True):
                continue
            dv = m.get_total_cost()
            if dv < min_dv:
                min_dv = dv
                best_mnv = m
        if best_mnv is None:
            return None
        return best_mnv
    
    @staticmethod
    def lambert(orb_v: Orbit, orb_t: Orbit, solver=bond, **kwargs):
        imps = lambert_planner(orb_v, orb_t, solver=solver, **kwargs)
        return Maneuver(imps, orb_v)

    @staticmethod
    def opt_bi_impulse_rdv(orbit_v: Orbit, 
                           orbit_t: Orbit, 
                           safety_check: bool = True,
                           before: float = None) -> Maneuver | None:
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
    def opt_lambert_multi_revolution(orbit_v: Orbit, 
                                     orbit_t: Orbit) -> Maneuver | None:
        """lambert问题最佳多周转解

        Args:
            orbit_v (Orbit): 初始轨道
            orbit_t (Orbit): 目标轨道

        Returns:
            Maneuver: 双脉冲转移机动
        """
        return opt_lambert_revolution(orbit_v, orbit_t)

    @classmethod
    def course_correction(cls, orb_v: Orbit, orb_t: Orbit, dt=5):
        orb_mnv_start = orb_v
        orb_mnv_end = orb_t.propagate(-dt)
        mnv = Maneuver.lambert(orb_mnv_start, orb_mnv_end)
        return mnv
        
    @classmethod
    def moon_transfer_target(cls, 
                             orb_v: Orbit, 
                             moon: Body, 
                             cap_t: float, 
                             pe: float = 100.,
                             inc: float = 0.,
                             relative: bool = True):
        """卫星转移瞄准轨道

        Args:
            orb_v (Orbit): 航天器轨道
            moon (Body): 卫星
            cap_t (float): 捕获时刻
            pe (float, optional): 捕获轨道的近星点高度. Defaults to 100*u.km.
            inc (float, optional): 捕获轨道倾角. Defaults to 0*u.rad.
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
                                   cap_t: float,
                                   rp_t: float | None = None
                                   ):
        """向卫星目标轨道转移的瞄准轨道, 位于捕获临界前

        Args:
            orb_v (Orbit): 航天器轨道
            orb_t (Orbit): 目标轨道
            cap_t (float): 瞄准捕获时间
            rp_m (float): 飞越近星点

        Returns:
            Orbit: 瞄准轨道
        """
        # 瞄准捕获时间当前只是开始搜索时间, 因此如果瞄准时间大于最近的窗口, 可能会错过较近的窗口
        orb_target = transfer_orbit_target(orb_v, orb_t, cap_t, rp_t=rp_t)
        orb_transfer = orb_target.propagate_to_nu(0, M=-1)
        if orb_transfer.epoch < orb_v.epoch:
            cap_t += orb_t.attractor.rotational_period / 2
            warnings.warn(f'orbit already passed transfer window, trying capture time {cap_t}', RuntimeWarning)
            return cls.moon_orbit_transfer_target(orb_v, orb_t, cap_t)
        return orb_target

    @classmethod
    def moon_return_target(cls, orb_v: Orbit, pe: float, esc_t: float):
        """从卫星返回行星指定近星点高度的瞄准轨道, 位于逃逸临界前

        Args:
            orb_v (Orbit): 航天器轨道
            pe (float): 近星点高度
            esc_t (float): 瞄准逃逸时间

        Returns:
            Orbit: 瞄准轨道
        """
        orb_target = return_target(orb_v, pe, esc_t)
        orb_transfer = orb_target.propagate_to_nu(0, M=-1)
        if orb_transfer.epoch < orb_v.epoch:
            esc_t += orb_v.attractor.rotational_period / 2
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
        orb_transfer    = orb_t.propagate_to_nu(0, M=-1)
        mnv_coplanar    = Maneuver.match_plane(orb_v, orb_transfer, closest=True, conserved=True)
        orb_coplanar    = mnv_coplanar.apply()
        delta_nu        = mv.angle_between_vectors(orb_coplanar.r_vec, orb_transfer.r_vec, orb_coplanar.h_vec)
        orb_coplanar    = orb_coplanar.propagate_to_nu(orb_coplanar.nu + delta_nu)
        print(f'revisit dt: {orb_transfer.epoch - orb_coplanar.epoch}')
        mnv_phase       = Maneuver.change_phase(orb_coplanar, orb_transfer.epoch, immediate=True, conserved=False)
        orb_start       = mnv_phase.apply()
        orb_start       = orb_start.propagate_to_epoch(orb_transfer.epoch)
        mnv_transfer    = Maneuver.lambert(orb_start, orb_t)
        mnv_series      = Maneuver.serial(orb_v, [mnv_coplanar, mnv_phase, mnv_transfer])
        return mnv_series
