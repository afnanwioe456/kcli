from __future__ import annotations
from astropy import units as u
from typing import TYPE_CHECKING

from .tasks import Task
from .maneuver import ExecuteNode
from ..astro.orbit import Orbit
from ..astro.maneuver import Maneuver
from ..utils import *

if TYPE_CHECKING:
    from .tasks import Tasks
    from ..spacecrafts import SpacecraftBase


class Transfer(Task):
    def __init__(self, 
                 spacecraft: SpacecraftBase, 
                 tasks: Tasks, 
                 orb_t: Orbit,
                 start_time: float = -1, 
                 duration: int = 300, 
                 importance: int = 6,
                 ):
        """瞄准轨道转移规划"""
        super().__init__(spacecraft, tasks, start_time, duration, importance)
        self.orb_t = orb_t

    @property
    def description(self):
        return (f'{self.spacecraft.name} 轨道转移规划\n'
                f'\t瞄准轨道: {self.orb_t}')

    @logging_around
    def start(self):
        if not self._conn_setup():
            return
        orb_v = Orbit.from_krpcv(self.vessel)
        mnv = Maneuver.transfer(orb_v, self.orb_t)
        nodes = mnv.to_krpcv(self.vessel)
        mnv_tasks = []
        for n in nodes[:-1]:
            task = ExecuteNode.from_node(
                self.spacecraft, 
                n, 
                self.tasks, 
                importance=self.importance)
            mnv_tasks.append(task)
        self.tasks.submit_nowait(mnv_tasks)
        orb_cor = self.orb_t.propagate_to_r(self.orb_t.a, M=-1)
        correct_task = CourseCorrect(
            self.spacecraft, 
            self.tasks, 
            self.orb_t, 
            start_time = orb_cor.epoch.to_value(u.s))
        self.tasks.submit(correct_task)
        self.conn.close()


class CourseCorrect(Task):
    def __init__(self, 
                 spacecraft: SpacecraftBase, 
                 tasks: Tasks, 
                 orb_t: Orbit,
                 start_time: float = -1, 
                 duration: int = 300, 
                 importance: int = 6,
                 ):
        """轨道修正"""
        super().__init__(spacecraft, tasks, start_time, duration, importance)
        self.orb_t = orb_t

    @property
    def description(self):
        return (f'{self.spacecraft.name} 轨道修正\n'
                f'\t瞄准轨道: {self.orb_t}')

    @logging_around
    def start(self):
        if not self._conn_setup():
            return
        time_wrap(self.start_time)
        orb_v = Orbit.from_krpcv(self.vessel)
        mnv = Maneuver.course_correction(orb_v, self.orb_t)
        print(mnv)
        nodes = mnv.to_krpcv(self.vessel)
        mnv_tasks = []
        for n in nodes:
            task = ExecuteNode.from_node(
                self.spacecraft, 
                n, 
                self.tasks,
                importance=self.importance)
            mnv_tasks.append(task)
        self.tasks.submit_nowait(mnv_tasks)
        self.conn.close()
    