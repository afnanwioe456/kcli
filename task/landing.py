from __future__ import annotations
from time import sleep
from krpc.services.spacecenter import SASMode

from .tasks import Task
from ..astro.orbit import Orbit
from ..astro.maneuver import Maneuver
from ..utils import *


if TYPE_CHECKING:
    from .tasks import Tasks
    from ..spacecrafts import Spacecraft
    from ..astro.body import Body


__all__ = [
    'GlideLanding',
]


class GlideLanding(Task):
    def __init__(self, 
                 spacecraft: Spacecraft, 
                 tasks: Tasks, 
                 start_time: float = -1, 
                 duration: float = 1800, 
                 importance: int = 3,
                 submit_next: bool = True):
        super().__init__(spacecraft, tasks, start_time, duration, importance, submit_next)

    @property
    def description(self):
        return (f'{self.spacecraft.name} -> 无动力着陆'
                f'\t预计执行时: {sec_to_date(self.start_time)}')

    @logging_around
    def start(self):
        if not self._conn_setup():
            return
        krpc_orb = self.vessel.orbit
        orbv = Orbit.from_krpcorb(krpc_orb)
        while orbv.pe > orbv.attractor.atmosphere_height:
            # 如果航天器此时还未转移到环绕要着陆的天体轨道上
            # FIXME: 判断过于简略, 如果已经进入大气层的情况
            krpc_orb = krpc_orb.next_orbit
            if krpc_orb is None:
                raise ValueError(f'{self.spacecraft.name} not on a reentry orbit.')
            orbv = Orbit.from_krpcorb(krpc_orb)
        orb_reentry = orbv.propagate_to_r(orbv.attractor.r + orbv.attractor.atmosphere_height, sign=False)
        time_wrap(orb_reentry.epoch)
        self.vessel = self.sc.active_vessel
        while not get_parts_in_stage_by_type(
            self.vessel, 
            'parachute', 
            self.vessel.control.current_stage):
            self.vessel.control.activate_next_stage()
            sleep(1)
            self.vessel = self.sc.active_vessel
        self.vessel.control.rcs = True
        self.vessel.control.sas = True
        sleep(0.1)
        self.vessel.control.sas_mode = SASMode.retrograde
        self.sc.physics_warp_factor = 3
        recover_stream = self.conn.add_stream(getattr, self.vessel, 'recoverable')
        mass_stream = self.conn.add_stream(getattr, self.vessel, 'mass')
        while not recover_stream() or mass_stream > 0:
            sleep(1)
        if self.vessel.recoverable:
            self.vessel.recover()
        dummy_roll_out()
        self.conn.close()


class Landing(Task):
    def __init__(self, 
                 spacecraft: Spacecraft, 
                 tasks: Tasks, 
                 body: Body,
                 landing_site: tuple[float],
                 start_time: float = -1, 
                 duration: float = 1800, 
                 importance: int = 3,
                 submit_next: bool = True):
        super().__init__(spacecraft, tasks, start_time, duration, importance, submit_next)
        self.body = body
        self.landing_site = landing_site

    @property
    def description(self):
        return (f'{self.spacecraft.name} -> 着陆 -> {self.body.name}'
                f'\t坐标: {self.landing_site}'
                f'\t预计执行时: {sec_to_date(self.start_time)}')

    
    