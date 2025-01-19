from __future__ import annotations
from typing import TYPE_CHECKING

from .tasks import Task
from .maneuver import ExecuteNode
from ..astro.orbit import Orbit
from ..astro.maneuver import Maneuver
from ..utils import sec_to_date, logging_around

if TYPE_CHECKING:
    from .tasks import Tasks
    from ..spacecrafts import SpaceStation
    from ..spacecrafts import SpacecraftBase


class Rendezvous(Task):
    def __init__(self,
                 spacecraft: SpacecraftBase,
                 spacestation: SpaceStation,
                 tasks: Tasks,
                 start_time=-1,
                 duration=300,
                 ):
        super().__init__(spacecraft, tasks, start_time, duration)
        self.spacestation = spacestation
        self.conn = None

    @property
    def description(self):
        return (f'{self.name} -> {self.spacestation.name} 交会规划\n'
                f'\t预计执行时: {sec_to_date(int(self.start_time))}')

    @logging_around
    def start(self):
        if not self._conn_setup():
            return
        ss_orb = Orbit.from_krpcv(self.spacestation.vessel)
        v_orb = Orbit.from_krpcv(self.vessel)
        mnv = Maneuver.opt_bi_impulse_rdv(v_orb, ss_orb)
        nodes = mnv.to_krpcv(self.vessel)
        next_task: list[Task] = []
        for n in nodes:
            task = ExecuteNode.from_node(self.spacecraft, n, self.tasks, importance=8)
            next_task.append(task)
        self.tasks.submit_nowait(next_task)
        
        