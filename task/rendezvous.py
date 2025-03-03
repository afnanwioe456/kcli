from __future__ import annotations
from typing import TYPE_CHECKING
from astropy import units as u

from .tasks import Task
from .maneuver import ExecuteNode
from ..astro.orbit import Orbit
from ..astro.maneuver import Maneuver
from ..utils import sec_to_date, logging_around

if TYPE_CHECKING:
    from .tasks import Tasks
    from ..spacecrafts import SpaceStation
    from ..spacecrafts import SpacecraftBase

__all__ = [
    'Rendezvous',
]


class Rendezvous(Task):
    def __init__(self,
                 spacecraft: SpacecraftBase,
                 spacestation: SpaceStation,
                 tasks: Tasks,
                 start_time: u.Quantity = -1 * u.s,
                 duration: u.Quantity = 300 * u.s,
                 importance: int = 6, 
                 ):
        super().__init__(spacecraft, tasks, start_time, duration, importance)
        self.spacestation = spacestation

    @property
    def description(self):
        return (f'{self.name} -> {self.spacestation.name} 交会规划\n'
                f'\t预计执行时: {sec_to_date(self.start_time)}')

    @logging_around
    def start(self):
        if not self._conn_setup():
            return
        # 非常奇怪的情况: 如果不先锁定目标再计算KSP就会有很大的误差
        # 问题似乎发生在部署机动节点的瞬间
        self.sc.target_vessel = self.spacestation.vessel
        ss_orb = Orbit.from_krpcv(self.spacestation.vessel)
        v_orb = Orbit.from_krpcv(self.vessel)
        mnv = Maneuver.opt_bi_impulse_rdv(v_orb, ss_orb)
        nodes = mnv.to_krpcv(self.vessel)
        next_task: list[Task] = []
        for n in nodes:
            task = ExecuteNode.from_node(self.spacecraft, n, self.tasks, importance=8)
            next_task.append(task)
        self.tasks.submit_nowait(next_task)
        self.conn.close()
        
    def _to_dict(self):
        dic = {
            'spacestation_name': self.spacestation.name,
        }
        return super()._to_dict() | dic
    
    @classmethod
    def _from_dict(cls, data, tasks):
        from ..spacecrafts import SpacecraftBase
        return cls(
            spacecraft = SpacecraftBase.get(data['spacecraft_name']),
            spacestation = SpacecraftBase.get(data['spacestation_name']),
            tasks = tasks,
            start_time = data['start_time'] * u.s,
            duration = data['duration'] * u.s,
            importance = data['importance'],
        )
        