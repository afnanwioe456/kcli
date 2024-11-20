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
                 step=None,
                 upper_stage=False,  # 是否有上面级需要分离
                 ):
        super().__init__(name, tasks, start_time, duration, step=step)
        self.importance = 7
        self.upper_stage = upper_stage
        self.executor = None

    def maneuver_prediction(self):
        """
        进行一次机动节点预测，将预计start_time交给manager修正
        """
        pass


class SimpleMnv(Maneuver):
    def __init__(self,
                 name,
                 tasks,
                 mode,
                 target,
                 start_time=-1,
                 duration=300,
                 step=None,
                 upper_stage=False):
        super().__init__(name, tasks, start_time, duration, step, upper_stage)
        self.mode = mode
        self.target = target
        self.conn = None
        self.corrected_flag = False  # 执行时间是否被修正过
        self.nodes = []

    @property
    def description(self):
        if not self.corrected_flag:
            return (f'{self.name} 机动节点规划\n'
                    f'\t目标{self.mode}：{self.target}\n'
                    f'\t预计执行时：{sec_to_date(int(self.start_time))}')
        return (f'{self.name} 轨道机动\n'
                f'\t目标{self.mode}：{self.target}\n'
                f'\t预计执行时：{sec_to_date(int(self.start_time))}')

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
        else:
            print(f'不受{self.__class__.__name__}支持的机动模式: {self.mode}')
            return False
        return True

    def _maneuver_plan(self):
        self.nodes = self.vessel.control.nodes
        for node in self.nodes:
            node.remove()
        try:
            next_node = self.operation.make_node()
        except:
            log = (f"{self.name}: an error occurred in make_mode: \n"
                   f"{self.operation.error_message}")
            write_log(log)
            return False
        # TODO: make_nodes prob?
        self.start_time = next_node.ut - 900
        self.importance = 3
        self.duration = 1800
        # TODO: 点火时间未计算
        if not self.corrected_flag:
            self.corrected_flag = True
            self.tasks.submit_nowait(self)
        return True

    def start(self):
        if not self._conn_setup():
            return
        add_abort_callback(self)

        if not self.corrected_flag:
            if self._maneuver_plan():
                log = f"{self.name}: new maneuver node created"
                write_log(log)
            self.conn.close()
            return

        # TODO: 时间加速后节点漂移问题？submit后立即创建新节点？
        self.vessel.control.rcs = True
        self._maneuver_plan()
        self.executor = self.mj.node_executor
        self.executor.autowarp = True
        self.executor.tolerance = 0.1
        self.executor.execute_one_node()
        sleep(1)

        write_log(f"{self.name}: attitude adjustment")

        """
        if self.upper_stage:
            self.upper_stage_flameout_check()
        """

        while not self.tasks.abort_flag and self.executor.enabled:
            sleep(5)

        if self.tasks.abort_flag:
            write_log(f"{self.name}: abort")
            self.conn.close()

        self.conn.close()
        write_log(f"{self.name}: maneuver complete")


if __name__ == '__main__':
    from task.tasks import Tasks
    from command import ChatMsg
    from task.tasks import TaskQueue

    TASK_QUEUE = TaskQueue()
    msg = ChatMsg('', '', '', '', 0)
    tasks = Tasks(msg, 0, 3, TASK_QUEUE)
    # opr_ap = SimpleMnv('_Relay_High', tasks, 'ap', 600000, upper_stage=True)
    opr_pe = SimpleMnv("launch_test_2's flight", tasks, 'ap', 1600000, upper_stage=True)
    # opr_inc = SimpleMnv('launch.py的测试飞行', tasks, 'inc', upper_stage=True)

    # tasks.submit(opr_ap)
    tasks.submit(opr_pe)
    # tasks.submit(opr_inc)
    TASK_QUEUE.put(tasks)
    for i in range(6):
        tasks = TASK_QUEUE.get()
        print('----------')
        sleep(3)
        tasks = tasks.do()
        if tasks:
            TASK_QUEUE.submit(tasks)
