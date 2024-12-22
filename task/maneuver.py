from __future__ import annotations
from time import sleep
from astropy import units as u

from .tasks import Task
from ..astro.orbit import *
from ..astro.maneuver import *
from ..utils import *

if TYPE_CHECKING:
    from task.tasks import Tasks
    from krpc.services.spacecenter import Vessel, Node


class SimpleMnvPlan(Task):
    def __init__(self,
                 name: str,
                 tasks: Tasks,
                 mode: str,
                 target: u.Quantity,
                 start_time: float = -1,
                 duration: int = 300,
                 importance: int = 3,
                 ):
        """规划一次简单的轨道机动

        Args:
            name (str): 载具名
            tasks (Tasks): Tasks对象
            mode (str): 机动规划模式(ap, pe, inc)
            target (float): 规划机动的目标
            start_time (float, optional): 任务执行时. Defaults to -1.
            duration (int, optional): 任务时长. Defaults to 300.
        """
        super().__init__(name, tasks, start_time, duration, importance)
        self.mode = mode
        self.target = target

    @property
    def description(self):
        return (f'{self.name} 机动节点规划\n'
                f'\t目标{self.mode}: {self.target}\n'
                f'\t预计执行时: {sec_to_date(self.start_time)}')

    def _conn_setup(self):
        if not switch_to_vessel(self.name):
            return False
        if not super()._conn_setup(f'mnv: {self.name}'):
            return False
        return True

    @staticmethod
    def _invoke_mnv(mode, *args):
        if mode == 'ap':
            return Maneuver.change_apoapsis(*args)
        if mode == 'pe':
            return Maneuver.change_periapsis(*args)
        if mode == 'inc':
            orb_v, inc = args
            orb_t = Orbit.from_coe(orb_v.attractor, orb_v.a, orb_v.e, inc,
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
            task_list.append(ExecuteNode.from_node(self.vessel, n, self.tasks))
        self.tasks.submit_nowait(task_list)
        self.conn.close()


class ExecuteNode(Task):
    def __init__(self, name, tasks, start_time, duration, importance = 7):
        """执行最近的一个节点"""
        super().__init__(name, tasks, start_time, duration, importance)

    @property
    def description(self):
        return (f'{self.name} 轨道机动\n'
                f'\t预计执行时: {sec_to_date(self.start_time)}')
        
    def _conn_setup(self):
        if not switch_to_vessel(self.name):
            return False
        if not super()._conn_setup(f'mnv: {self.name}'):
            return False
        return True

    @logging_around
    def start(self):
        if not self._conn_setup():
            return
        self.vessel.control.rcs = True
        self.executor = self.mj.node_executor
        self.executor.autowarp = True
        self.executor.tolerance = 0.1
        self.executor.execute_one_node()
        while not self.tasks.abort_flag and self.executor.enabled:
            sleep(5)
        self.conn.close()

    @staticmethod
    def from_node(vessel: Vessel, node: Node, tasks: Tasks, importance=7):
        """执行节点"""
        # TODO: duration
        return ExecuteNode(vessel.name, tasks, node.ut, 1800, importance)