from __future__ import annotations
from typing import TYPE_CHECKING

from .tasks import Task
from .maneuver import SimpleMnv
from ..astro.orbit import Orbit
from ..astro.maneuver import Maneuver as MnvPlanner
from ..utils import sec_to_date, switch_to_vessel, get_vessel_by_name, LOGGER, logging_around

if TYPE_CHECKING:
    from .tasks import Tasks
    from ..spacecrafts import Spacestation


class Rendezvous(Task):
    def __init__(self,
                 name: str,
                 spacestation: Spacestation,
                 tasks: Tasks,
                 start_time=-1,
                 duration=300,
                 ):
        super().__init__(name, tasks, start_time, duration)
        self.spacestation = spacestation
        self.conn = None

    @property
    def description(self):
        return (f'{self.name} -> {self.spacestation.name} 交会规划\n'
                f'\t预计执行时: {sec_to_date(int(self.start_time))}')

    def _conn_setup(self):
        if not switch_to_vessel(self.name):
            return False
        if not super()._conn_setup(f'rdv: {self.name}'):
            return False
        return True

    @logging_around
    def start(self):
        if not self._conn_setup:
            return
        v = get_vessel_by_name(self.name)
        ss_orb = Orbit.from_krpcv(self.spacestation.vessel)
        v_orb = Orbit.from_krpcv(v)
        mnv = MnvPlanner.bi_impulse(v_orb, ss_orb)
        nodes = mnv.to_krpcv(v)
        next_task: list[Task] = []
        for n in nodes:
            start_time = n.ut - 600
            task = SimpleMnv(self.name, self.tasks, 'node', start_time=start_time)
            task.importance = 8
            next_task.append(task)
        self.tasks.submit_nowait(next_task)
        
        