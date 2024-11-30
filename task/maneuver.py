from __future__ import annotations
from time import sleep

from .tasks import Task
from ..utils import *

if TYPE_CHECKING:
    from task.tasks import Tasks


class Maneuver(Task):
    def __init__(self,
                 name: str,
                 tasks: Tasks,
                 start_time=-1,
                 duration=300,
                 ):
        super().__init__(name, tasks, start_time, duration)
        self.importance = 7
        self.executor = None

    def maneuver_prediction(self):
        """
        进行一次机动节点预测，将预计start_time交给manager修正
        """
        pass


class SimpleMnv(Maneuver):
    def __init__(self,
                 name: str,
                 tasks: Tasks,
                 mode: str,
                 target: float = 0,
                 start_time: int = -1,
                 duration: int = 300,
                 ):
        """规划一次简单的轨道机动, 或者执行一个机动节点

        Args:
            name (str): 载具名
            tasks (Tasks): Tasks对象
            mode (str): 机动规划模式(ap, pe, inc), node直接执行一个节点
            target (float): 当模式置于(ap, pe, inc)时, 规划机动的目标
            start_time (int, optional): 任务执行时. Defaults to -1.
            duration (int, optional): 任务时长. Defaults to 300.
        """
        super().__init__(name, tasks, start_time, duration)
        self.mode = mode
        self.target = target
        self.conn = None
        if self.mode == 'node': 
            self.corrected_flag = True
        else:
            self.corrected_flag = False
        self.nodes = []

    @property
    def description(self):
        if not self.corrected_flag:
            return (f'{self.name} 机动节点规划\n'
                    f'\t目标{self.mode}: {self.target}\n'
                    f'\t预计执行时: {sec_to_date(int(self.start_time))}')
        if self.mode == 'node':
            return (f'{self.name} 轨道机动\n'
                    f'\t预计执行时: {sec_to_date(int(self.start_time))}')
        return (f'{self.name} 轨道机动\n'
                f'\t目标{self.mode}: {self.target}\n'
                f'\t预计执行时: {sec_to_date(int(self.start_time))}')

    def _conn_setup(self, **kwargs):
        if not switch_to_vessel(self.name):
            return False
        super()._conn_setup(f'mnv: {self.name}')
        self.step = self.vessel.orbit.period
        if self.mode == 'ap':
            self.operation = self.mj.maneuver_planner.operation_apoapsis
            self.operation.new_apoapsis = self.target
        elif self.mode == 'pe':
            self.operation = self.mj.maneuver_planner.operation_periapsis
            self.operation.new_periapsis = self.target
        elif self.mode == 'inc':
            self.operation = self.mj.maneuver_planner.operation_inclination
            self.operation.new_inclination = self.target
        elif self.mode != 'node':
            LOGGER.debug(f'不受{self.__class__.__name__}支持的机动模式: {self.mode}')
            return False
        return True

    def _maneuver_plan(self):
        self.nodes = self.vessel.control.remove_nodes()
        try:
            next_node = self.operation.make_node()
        except:
            log = (f"{self.name}: an error occurred in make_mode: \n"
                   f"{self.operation.error_message}")
            LOGGER.debug(log)
            return False
        # TODO: make_nodes prob?
        self.start_time = next_node.ut - 600
        self.duration = 1200
        # TODO: 点火时间未计算
        if not self.corrected_flag:
            self.corrected_flag = True
            self.tasks.submit_nowait(self)
        return True

    @logging_around
    def start(self):
        if not self._conn_setup():
            return
        if not self.corrected_flag:
            if self._maneuver_plan():
                log = f"{self.name}: new maneuver node created"
                LOGGER.debug(log)
            self.conn.close()
            return
        self.vessel.control.rcs = True
        self.executor = self.mj.node_executor
        self.executor.autowarp = True
        self.executor.tolerance = 0.1
        self.executor.execute_one_node()
        
        while not self.tasks.abort_flag and self.executor.enabled:
            sleep(5)
        self.conn.close()
