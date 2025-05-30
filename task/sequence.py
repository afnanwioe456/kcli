from __future__ import annotations
import numpy as np

from .tasks import Task
from ..astro.orbit import Orbit
from ..astro.body import Body
from ..astro.maneuver import Maneuver
from ..math import vector as mv
from ..utils import *

if TYPE_CHECKING:
    from .tasks import Tasks
    from ..spacecrafts import Spacecraft


class TaskSequence:
    """辅助类, 用于模块化创建任务序列"""
    @staticmethod
    def launch_to_lunar_orbit(
        spacecraft: Spacecraft,
        tasks: Tasks,
        rocket: str,
        payload: str,
        orbit: Orbit = None,
        pe_m: float = None,
        ap_m: float = None,
        inc_m: float = None
    ) -> list[Task]:
        epoch = get_ut()
        earth = Body.get_or_create('Earth')
        moon = Body.get_or_create('Moon')
        orb_v = Orbit.from_coe(
            earth, earth.r + 200000, 0, 0, 0, 0, 0, epoch
        )
        if orbit is not None:
            orb_t = orbit
        else:
            # 猜测一个较近的窗口, 然后规划转移机动
            orb_m = moon.orbit(epoch)
            S = np.array([1, 0, 0], dtype=np.float64)
            raan = mv.angle_between_vectors(S, orb_m.r_vec, orb_m.h_vec)
            orb_t = Orbit.from_coe(
                moon, moon.r + pe_m, 0, inc_m, raan, 0, 0, epoch
            )
        orb_tgt = Maneuver.moon_orbit_transfer_target(
            orb_v, orb_t, epoch
        )
        # 建立发射瞄准轨道
        orb_w = orb_tgt.propagate_to_nu(0, M=-1)
        mnv_cir = Maneuver.change_apoapsis(orb_w, orb_w.pe, immediate=True)
        orb_w = mnv_cir.apply()
        print(orb_tgt.epoch, orb_w.epoch, UTIL_CONN.space_center.ut)
        print(f'transfer time: {(orb_tgt.epoch - orb_w.epoch) / 86400} days')
        print(f'launch time: {(orb_w.epoch - UTIL_CONN.space_center.ut) / 86400} days')
        print(f'launch orbit: {orb_w}')
        
        from .launch import LAUNCH_ROCKET_DIC
        from .transfer import Transfer
        from .maneuver import CaptureMnv
        executor = LAUNCH_ROCKET_DIC[rocket]
        launch_task = executor(
            spacecraft = spacecraft,
            orbit = orb_w,
            tasks = tasks,
            payload = payload
        )
        transfer_task = Transfer(
            spacecraft = spacecraft,
            tasks = tasks,
            orb_t = orb_tgt,
        )
        capture_task = CaptureMnv(
            spacecraft = spacecraft,
            tasks = tasks,
            body = moon,
            ap_t = ap_m,
        )
        return [launch_task, transfer_task, capture_task]
        
    @staticmethod
    def launch_to_lunar_surface(
        spacecraft: Spacecraft,
        tasks: Tasks,
        rocket: str,
        payload: str,
        landing_coord: tuple[float, float]
    ) -> list[Task]:
        task_list = TaskSequence.launch_to_lunar_orbit(
            spacecraft=spacecraft,
            tasks=tasks,
            rocket=rocket,
            payload=payload,
            pe_m=100000,
            ap_m=100000,
            inc_m=np.pi/2
        )
        moon = Body.get_or_create('Moon')

        from .landing import LandingMnv, Landing
        landing_mnv_task = LandingMnv(
            spacecraft=spacecraft,
            tasks=tasks,
            body=moon,
            landing_coord=landing_coord,
            deorbit_alt=-moon.r/10,
        )
        landing_task = Landing(
            spacecraft=spacecraft,
            tasks=tasks,
            landing_coord=landing_coord
        )
        return task_list.extend([landing_mnv_task, landing_task])
