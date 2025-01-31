from __future__ import annotations
from time import sleep
from krpc.services.spacecenter import SASMode
from astropy import units as u

from .tasks import Task
from ..astro.orbit import Orbit
from ..utils import *


if TYPE_CHECKING:
    from .tasks import Tasks
    from ..spacecrafts import SpacecraftBase

__all__ = [
    'GlideLanding',
    'ControlledReentry',
]


class GlideLanding(Task):
    def __init__(self, 
                 spacecraft: SpacecraftBase, 
                 tasks: Tasks, 
                 start_time: float = -1, 
                 duration: int = 1800, 
                 importance: int = 3,
                 ):
        super().__init__(spacecraft, tasks, start_time, duration, importance)

    @property
    def description(self):
        return (f'{self.name} -> 无动力着陆'
                f'  预计执行时: {sec_to_date(self.start_time)}')

    @logging_around
    def start(self):
        if not self._conn_setup():
            return
        krpc_orb = self.vessel.orbit
        orbv = Orbit.from_krpcorb(krpc_orb)
        while orbv.pe > orbv.attractor.atomsphere_height:
            # 如果航天器此时还未转移到环绕要着陆的天体轨道上
            krpc_orb = krpc_orb.next_orbit
            if krpc_orb is None:
                raise ValueError(f'{self.name} not on a reentry orbit.')
            orbv = Orbit.from_krpcorb(krpc_orb)
        orb_reentry = orbv.propagate_to_r(orbv.attractor.r + orbv.attractor.atomsphere_height, sign=False)
        time_wrap(orb_reentry.epoch.to_value(u.s))
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
        while not recover_stream():
            sleep(1)
        self.vessel.recover()
        dummy_roll_out()
        self.conn.close()

        
class ControlledReentry(Task):
    def __init__(self, 
                 spacecraft: SpacecraftBase, 
                 tasks: Tasks, 
                 start_time: float = -1, 
                 duration: int = 1800, 
                 importance: int = 3,
                 ):
        super().__init__(spacecraft, tasks, start_time, duration, importance)

    @property
    def description(self):
        return (f'{self.name} -> 受控再入'
                f'  预计执行时: {sec_to_date(self.start_time)}')

    @logging_around
    def start(self):
        if not self._conn_setup():
            return
        krpc_orb = self.vessel.orbit
        orbv = Orbit.from_krpcorb(krpc_orb)
        while orbv.pe > orbv.attractor.atomsphere_height:
            krpc_orb = krpc_orb.next_orbit
            if krpc_orb is None:
                raise ValueError(f'{self.name} not on a reentry orbit.')
            orbv = Orbit.from_krpcorb(krpc_orb)
        orb_reentry = orbv.propagate_to_r(orbv.attractor.r + orbv.attractor.atomsphere_height, sign=False)
        time_wrap(orb_reentry.epoch.to_value(u.s))
        self.sc.physics_warp_factor = 3
        mass_stream = self.conn.add_stream(getattr, self.vessel, 'mass')
        while mass_stream() > 0:
            sleep(1)
        dummy_roll_out()
        self.conn.close()