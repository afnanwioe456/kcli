from __future__ import annotations
from typing import TYPE_CHECKING
from time import sleep

from .tasks import Task
from ..autopilot import get_closer
from ..utils import LOGGER, switch_to_vessel, sec_to_date, logging_around

if TYPE_CHECKING:
    from krpc.client import Client
    from .tasks import Tasks
    from ..spacecrafts import Spacestation


def docking_with_target(conn: Client, ss: Spacestation):
    conn.space_center.target_vessel = ss.vessel
    sc = conn.space_center
    mj = conn.mech_jeb
    active = sc.active_vessel
    sleep(5)

    parts = active.parts
    # TODO: 对接口的选择问题
    parts.controlling = parts.docking_ports[0].part
    sc.target_docking_port = ss.crew_docking_ports[0]

    log = f"{conn.space_center.active_vessel.name}: docking with {ss.name}"
    LOGGER.debug(log)
    docking = mj.docking_autopilot
    docking.enabled = True

    while docking.enabled:
        sleep(1)


class Docking(Task):
    def __init__(self, 
                 name: str, 
                 spacestation: Spacestation,
                 tasks: Tasks, 
                 start_time: int = -1, 
                 duration: int = 1800, 
                 importance: int = 3):
        super().__init__(name, tasks, start_time, duration, importance)
        self.spacestation = spacestation

    @property
    def description(self):
        return (f'{self.name} -> {self.spacestation.name} 对接\n'
                f'\t预计执行时: {sec_to_date(int(self.start_time))}')

    def _conn_setup(self):
        if not switch_to_vessel(self.name):
            return False
        if not super()._conn_setup(f'docking: {self.name}'):
            return False
        return True
    
    @logging_around
    def start(self):
        if not self._conn_setup():
            return False
        # TODO: 主引擎与RCS控制
        dis = 50
        LOGGER.debug(f'{self.name} -> {self.spacestation.name} getting closer: {dis}')
        get_closer(dis, self.vessel, self.spacestation.vessel)
        LOGGER.debug(f'{self.name} -> {self.spacestation.name} docking')
        docking_with_target(self.conn, self.spacestation)
        LOGGER.debug(f'{self.name} -> {self.spacestation.name} docking completed')
        self.conn.close()
