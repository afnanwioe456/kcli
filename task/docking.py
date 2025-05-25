from __future__ import annotations
from typing import TYPE_CHECKING
from time import sleep

from .tasks import Task
from ..autopilot import get_closer
from ..utils import *

if TYPE_CHECKING:
    from krpc.client import Client
    from .tasks import Tasks
    from ..spacecrafts import *

__all__ = [
    'Docking',
    'Undocking',
]


def _dock_with_target(conn: Client, ss: SpaceStation, docking_port: DockingPortExt):
    conn.space_center.target_vessel = ss.vessel
    sc = conn.space_center
    mj = conn.mech_jeb
    active = sc.active_vessel
    parts = active.parts
    parts.controlling = parts.docking_ports[0].part
    sc.target_docking_port = docking_port.part.docking_port

    docking = mj.docking_autopilot
    docking.speed_limit = 10
    docking.enabled = True

    while docking.enabled:
        sleep(1)


class Docking(Task):
    def __init__(self, 
                 spacecraft: Spacecraft, 
                 spacestation: SpaceStation,
                 tasks: Tasks, 
                 start_time: float = -1, 
                 duration: float = 1800,
                 importance: int = 3,
                 submit_next: bool = True):
        super().__init__(spacecraft, tasks, start_time, duration, importance, submit_next)
        self.spacestation = spacestation

    @property
    def description(self):
        return (f'{self.spacecraft.name} -> 对接 -> {self.spacestation.name}\n'
                f'\t预计执行时: {sec_to_date(self.start_time)}')

    @logging_around
    def start(self):
        UTIL_CONN.space_center.target_vessel = self.spacestation.vessel
        target_dpext = self.spacestation.part_exts.get_target_docking_port(
            self.spacecraft.part_exts.active_docking_port_ext)
        if target_dpext.is_docked():
            LOGGER.debug(f'{self.spacestation} docking port [{target_dpext.part_id}] '
                         f'is docked with {target_dpext.docked_with}, initializing return task.')
            return_tasks = self.spacestation.return_mission(target_dpext, self.tasks)
            self.tasks.submit_nowait(return_tasks[0] + [self] + return_tasks[1:])
            return
        if not self._conn_setup():
            return
        dis = 50
        self.spacecraft.part_exts.rcs = True
        self.spacecraft.part_exts.main_engines = True
        LOGGER.debug(f'{self.spacecraft.name} -> {self.spacestation.name} getting closer: {dis}')
        get_closer(dis, self.vessel, self.spacestation.vessel)
        LOGGER.debug(f'{self.spacecraft.name} -> {self.spacestation.name} docking')
        _dock_with_target(self.conn, self.spacestation, target_dpext)
        LOGGER.debug(f'{self.spacecraft.name} -> {self.spacestation.name} docking completed')
        self.conn.close()
    
    def _to_dict(self):
        dic = {
            'spacestation_name': self.spacestation.name,
            }
        return super()._to_dict() | dic

    @classmethod
    def _from_dict(cls, data, tasks):
        from ..spacecrafts import Spacecraft
        ss: SpaceStation = Spacecraft.get(data['spacestation_name'])
        return cls(
            spacecraft      = Spacecraft.get(data['spacecraft_name']),
            spacestation    = ss,
            tasks           = tasks,
            start_time      = data['start_time'],
            duration        = data['duration'],
            importance      = data['importance'],
            submit_next     = data['submit_next']
        )


class Undocking(Task):
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
        return (f'{self.spacecraft.docked_with} -> 对接分离 -> {self.spacecraft}\n'
                f'\t预计执行时: {sec_to_date(self.start_time)}')

    @logging_around
    def start(self):
        if not self._conn_setup():
            return False
        self.spacecraft.undock()
        sleep(3)
        v = self.spacecraft.vessel
        self.conn.space_center.active_vessel = v
        v.control.rcs = True
        v.control.forward = -0.5
        sleep(5)
        v.control.forward = 0
        v.control.rcs = False
        self.conn.close()
