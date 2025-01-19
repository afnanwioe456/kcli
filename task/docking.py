from __future__ import annotations
from typing import TYPE_CHECKING
from time import sleep

from .tasks import Task
from ..autopilot import get_closer
from ..utils import LOGGER, switch_to_vessel, sec_to_date, logging_around

if TYPE_CHECKING:
    from krpc.client import Client
    from .tasks import Tasks
    from ..spacecrafts import *


def docking_with_target(conn: Client, ss: SpaceStation, docking_port: DockingPortStatus):
    conn.space_center.target_vessel = ss.vessel
    sc = conn.space_center
    mj = conn.mech_jeb
    active = sc.active_vessel
    parts = active.parts
    parts.controlling = parts.docking_ports[0].part
    sc.target_docking_port = docking_port.part

    docking = mj.docking_autopilot
    docking.enabled = True

    while docking.enabled:
        sleep(1)


class Docking(Task):
    def __init__(self, 
                 spacecraft: SpacecraftBase, 
                 spacestation: SpaceStation,
                 docking_port: DockingPortStatus,
                 tasks: Tasks, 
                 start_time: int = -1, 
                 duration: int = 1800, 
                 importance: int = 3):
        super().__init__(spacecraft, tasks, start_time, duration, importance)
        self.spacestation = spacestation
        self.docking_port = docking_port

    @property
    def description(self):
        return (f'{self.name} -> {self.spacestation.name} 对接\n'
                f'\t预计执行时: {sec_to_date(int(self.start_time))}')

    @logging_around
    def start(self):
        if self.docking_port.is_docked():
            LOGGER.debug(f'{self.spacestation} docking port [{self.docking_port.num}] is docked with {self.docking_port.docked_with},'
                         f'initializing return task.')
            self.tasks.submit_nowait(self.docking_port.spacecraft.return_mission(self.docking_port, self.tasks) + [self])
            return
        if not self._conn_setup():
            return False
        # TODO: 主引擎与RCS控制
        dis = 50
        self.vessel.control.rcs = True
        LOGGER.debug(f'{self.name} -> {self.spacestation.name} getting closer: {dis}')
        get_closer(dis, self.vessel, self.spacestation.vessel)
        LOGGER.debug(f'{self.name} -> {self.spacestation.name} docking')
        docking_with_target(self.conn, self.spacestation, self.docking_port)
        LOGGER.debug(f'{self.name} -> {self.spacestation.name} docking completed')
        self.vessel.control.rcs = False
        self.conn.close()


class Undocking(Task):
    def __init__(self, 
                 spacestation: SpaceStation,
                 docking_port: DockingPortStatus,
                 tasks: Tasks, 
                 start_time: int = -1, 
                 duration: int = 1800, 
                 importance: int = 3):
        super().__init__(spacestation, tasks, start_time, duration, importance)
        self.spacestation = spacestation
        self.docking_port = docking_port
    
    @property
    def description(self):
        return (f'{self.spacestation}对接口[{self.docking_port.num}] -> 对接分离\n'
                f'\t预计执行时: {sec_to_date(int(self.start_time))}')

    @logging_around
    def start(self):
        if not self._conn_setup():
            return False
        s = self.docking_port.undock()
        self.spacecraft = s
        sleep(3)
        v = s.vessel
        self.conn.space_center.active_vessel = v
        v.control.rcs = True
        v.control.forward = -0.1
        sleep(5)
        v.control.forward = 0
        v.control.rcs = False
        self.conn.close()

        
