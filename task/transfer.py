from __future__ import annotations
from typing import TYPE_CHECKING

from .tasks import Task
from .maneuver import ExecuteNode
from ..astro.orbit import Orbit
from ..astro.maneuver import Maneuver
from ..utils import *

if TYPE_CHECKING:
    from .tasks import Tasks
    from ..spacecrafts import Spacecraft


class Transfer(Task):
    def __init__(self, 
                 spacecraft: Spacecraft, 
                 tasks: Tasks, 
                 orb_t: Orbit,
                 start_time: float = -1, 
                 duration: float = 300, 
                 importance: int = 6,
                 submit_next: bool = True,
                 ):
        """瞄准轨道转移规划"""
        super().__init__(spacecraft, tasks, start_time, duration, importance, submit_next)
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
        if self.submit_next:
            self._submit_next_task(nodes)

    def _submit_next_task(self, nodes):
        mnv_tasks = []
        for n in nodes[:-1]:
            task = ExecuteNode.from_node(
                self.spacecraft, 
                n, 
                self.tasks, 
                importance=self.importance)
            mnv_tasks.append(task)
        orb_cor = self.orb_t.propagate_to_r(self.orb_t.a, M=-1)
        correct_task = CourseCorrect(
            self.spacecraft, 
            self.tasks, 
            # FIXME: 提前一段时间机动, 否则在soi转移时会卡住
            self.orb_t.propagate(-3600),
            start_time = orb_cor.epoch)
        mnv_tasks.append(correct_task)
        self.tasks.submit_nowait(mnv_tasks)

    def _to_dict(self):
        dic =  {
            'orb_t': self.orb_t._to_dict()
        }
        return super()._to_dict() | dic
    
    @classmethod
    def _from_dict(cls, data, tasks):
        from ..spacecrafts import Spacecraft
        from ..astro.orbit import Orbit
        return cls(
            spacecraft  = Spacecraft.get(data['spacecraft_name']),
            tasks       = tasks,
            orb_t       = Orbit._from_dict(data['orb_t']),
            start_time  = data['start_time'],
            duration    = data['duration'],
            importance  = data['importance'],
            submit_next = data['submit_next'],
        )


class CourseCorrect(Task):
    def __init__(self, 
                 spacecraft: Spacecraft, 
                 tasks: Tasks, 
                 orb_t: Orbit,
                 start_time: float = -1, 
                 duration: float = 300, 
                 importance: int = 6,
                 submit_next: bool = True,
                 ):
        """轨道修正"""
        super().__init__(spacecraft, tasks, start_time, duration, importance, submit_next)
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
        # FIXME: 提前一点时间规划机动
        orb_v = Orbit.from_krpcv(self.vessel).propagate(60)
        mnv = Maneuver.course_correction(orb_v, self.orb_t)
        print(mnv)
        nodes = mnv.to_krpcv(self.vessel)
        if self.submit_next:
            self._submit_next_task(nodes)

    def _submit_next_task(self, nodes):
        mnv_tasks = []
        for n in nodes:
            task = ExecuteNode.from_node(
                self.spacecraft, 
                n, 
                self.tasks,
                importance=self.importance)
            mnv_tasks.append(task)
        self.tasks.submit_nowait(mnv_tasks)
    
    def _to_dict(self):
        dic =  {
            'orb_t': self.orb_t._to_dict()
        }
        return super()._to_dict() | dic
    
    @classmethod
    def _from_dict(cls, data, tasks):
        from ..spacecrafts import Spacecraft
        from ..astro.orbit import Orbit
        return cls(
            spacecraft  = Spacecraft.get(data['spacecraft_name']),
            tasks       = tasks,
            orb_t       = Orbit._from_dict(data['orb_t']),
            start_time  = data['start_time'],
            duration    = data['duration'],
            importance  = data['importance'],
            submit_next = data['submit_next'],
        )
