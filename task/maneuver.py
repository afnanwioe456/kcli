from __future__ import annotations
from time import sleep
from astropy import units as u

from .tasks import Task
from ..astro.orbit import *
from ..astro.maneuver import *
from ..utils import *

if TYPE_CHECKING:
    from .tasks import Tasks
    from ..spacecrafts import SpacecraftBase
    from krpc.services.spacecenter import Vessel, Node

__all__ = [
    'SimpleMnv',
    'ExecuteNode',
]


class SimpleMnv(Task):
    def __init__(self,
                 spacecraft: SpacecraftBase,
                 tasks: Tasks,
                 mode: str,
                 target: u.Quantity,
                 start_time: float = -1,
                 duration: int = 300,
                 importance: int = 3,
                 tol: float = 0.1,
                 ):
        """进行一次简单的轨道机动

        Args:
            name (str): 载具名
            tasks (Tasks): Tasks对象
            mode (str): 机动规划模式(ap, pe, inc)
            target (float): 规划机动的目标
            start_time (float, optional): 任务执行时. Defaults to -1.
            duration (int, optional): 任务时长. Defaults to 300.
        """
        super().__init__(spacecraft, tasks, start_time, duration, importance)
        self.mode = mode
        self.target = target
        self.tol = tol

    @property
    def description(self):
        return (f'{self.name} 机动节点规划\n'
                f'\t目标{self.mode}: {self.target}\n'
                f'\t预计执行时: {sec_to_date(self.start_time)}')

    @staticmethod
    def _invoke_mnv(mode, orb_v: Orbit, target: u.Quantity):
        if mode == 'ap':
            return Maneuver.change_apoapsis(orb_v, target)
        if mode == 'pe':
            return Maneuver.change_periapsis(orb_v, target)
        if mode == 'inc':
            orb_t = Orbit.from_coe(orb_v.attractor, orb_v.a, orb_v.e, target,
                                   orb_v.raan, orb_v.argp, orb_v.nu, orb_v.epoch)
            return Maneuver.match_plane(orb_v, orb_t, True)
        raise ValueError(f'maneuver mode: {mode}')

    @logging_around
    def start(self):
        if not self._conn_setup():
            return
        orbv = Orbit.from_krpcv(self.vessel)
        mnv = self._invoke_mnv(self.mode, orbv, self.target)
        nodes = mnv.to_krpcv(self.vessel)
        task_list = []
        for n in nodes:
            task_list.append(ExecuteNode.from_node(self.spacecraft, n, self.tasks, tol=self.tol))
        self.tasks.submit_nowait(task_list)
        self.conn.close()

    def _to_dict(self):
        dic = {
            'mode': self.mode,
            'target': self.target.to_value(u.deg) if self.mode == 'inc' else self.target.to_value(u.km),
            'tol': self.tol,
        }
        return super()._to_dict() | dic

    @classmethod
    def _from_dict(cls, data, tasks):
        from ..spacecrafts import SpacecraftBase
        mode = data['mode']
        return cls(
            spacecraft = SpacecraftBase.get(data['spacecraft_name']),
            tasks = tasks,
            mode = mode,
            target = data['target'] * u.deg if mode == 'inc' else data['target'] * u.km,
            start_time = data['start_time'],
            duration = data['duration'],
            importance = data['importance'],
            tol = data['tol'],
        )


class ExecuteNode(Task):
    def __init__(self, 
                 spacecraft: SpacecraftBase, 
                 tasks: Tasks, 
                 start_time: float, 
                 duration: int, 
                 importance: int = 7,
                 tol: float = 0.1, 
                 ):
        """执行最近的一个节点"""
        super().__init__(spacecraft, tasks, start_time, duration, importance)
        self.tol = tol

    @property
    def description(self):
        return (f'{self.name} 轨道机动\n'
                f'\t预计执行时: {sec_to_date(self.start_time)}')
        
    @logging_around
    def start(self):
        if not self._conn_setup():
            return
        self.vessel.control.rcs = True
        self.executor = self.mj.node_executor
        self.executor.autowarp = True
        self.executor.tolerance = self.tol
        self.executor.execute_one_node()
        while not self.tasks.abort_flag and self.executor.enabled:
            sleep(5)
        self.conn.close()

    @staticmethod
    def from_node(spacecraft: SpacecraftBase, node: Node, tasks: Tasks, tol=0.1, importance=7):
        """执行节点"""
        # TODO: duration
        return ExecuteNode(spacecraft, tasks, node.ut - 900, 1800, tol, importance)

    def _to_dict(self):
        dic = {
            'tol': self.tol,
        }
        return super()._to_dict() | dic
    
    @classmethod
    def _from_dict(cls, data, tasks):
        from ..spacecrafts import SpacecraftBase
        return cls(
            spacecraft = SpacecraftBase.get(data['spacecraft_name']),
            tasks = tasks,
            start_time = data['start_time'],
            duration = data['duration'],
            importance = data['importance'],
            tol = data['tol'],
        )