from __future__ import annotations
from time import sleep
from task.tasks import Task
from utils import *

if TYPE_CHECKING:
    from task.tasks import Tasks
    from spacecrafts import Spacestation


def rendezvous_planner(conn: Client, ss: Spacestation):
    conn.space_center.target_vessel = ss.vessel
    mj = conn.mech_jeb

    print("Planning Hohmann transfer")
    planner = mj.maneuver_planner
    hohmann = planner.operation_transfer
    hohmann.make_nodes()

    warning = hohmann.error_message
    if warning:
        print(warning)
        return False
    return True


def execute_nodes(executor):
    print("Executing maneuver nodes")
    executor.autowrap = True
    executor.execute_all_nodes()

    while executor.enabled:
        sleep(1)


def match_plane_with_target(conn: Client, ss: Spacestation):
    conn.space_center.target_vessel = ss.vessel
    mj = conn.mech_jeb
    planner = mj.maneuver_planner
    executor = mj.node_executor

    log = f"{conn.space_center.active_vessel.name}: matching plane with {ss.name}"
    write_log(log)
    match_plane = planner.operation_plane
    match_plane.make_nodes()
    execute_nodes(executor)


def rendezvous_with_target(conn: Client, ss: Spacestation):
    conn.space_center.target_vessel = ss.vessel
    mj = conn.mech_jeb
    planner = mj.maneuver_planner
    executor = mj.node_executor
    executor.autowrap = True

    execute_nodes(executor)

    log = f"{conn.space_center.active_vessel.name}: correcting course"
    write_log(log)
    fine_tune_closest_approach = planner.operation_course_correction
    fine_tune_closest_approach.intercept_distance = 1000
    fine_tune_closest_approach.make_nodes()
    executor.tolerance = 0.5
    execute_nodes(executor)

    log = f"{conn.space_center.active_vessel.name}: matching speed with {ss.name}"
    write_log(log)
    match_speed = planner.operation_kill_rel_vel
    match_speed.time_selector.time_reference = mj.TimeReference.closest_approach  # match speed at the closest approach
    match_speed.make_nodes()
    executor.tolerance = 0.5  # return the precision back to normal
    execute_nodes(executor)

    log = f"{conn.space_center.active_vessel.name}: rendezvous with {ss.name} completed"
    write_log(log)


def final_approach(conn: Client, ss: Spacestation):
    conn.space_center.target_vessel = ss.vessel
    mj = conn.mech_jeb
    autopilot = mj.rendezvous_autopilot
    autopilot.desired_distance = 100
    executor = mj.node_executor
    executor.autowrap = True

    log = f"{conn.space_center.active_vessel.name}: final approach to {ss.name}"
    write_log(log)
    autopilot.enabled = True

    while autopilot.enabled:
        sleep(1)


def docking_with_target(conn: Client, ss: Spacestation):
    conn.space_center.target_vessel = ss.vessel
    sc = conn.space_center
    mj = conn.mech_jeb
    active = sc.active_vessel
    sleep(5)

    parts = active.parts
    parts.controlling = parts.docking_ports[0].part
    sc.target_docking_port = ss.crew_docking_ports[0]

    log = f"{conn.space_center.active_vessel.name}: docking with {ss.name}"
    write_log(log)
    docking = mj.docking_autopilot
    docking.enabled = True

    while docking.enabled:
        sleep(1)


class Rendezvous_and_Docking(Task):
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
        self.plan_flag = False  # 是否进行过交会对接规划

    @property
    def description(self):
        if not self.plan_flag:
            return (f'{self.name} -> {self.spacestation.name} 交会对接规划\n'
                    f'\t预计执行时：{sec_to_date(int(self.start_time))}')
        return (f'{self.name} -> {self.spacestation.name} 交会对接\n'
                f'\t预计执行时：{sec_to_date(int(self.start_time))}')

    def _conn_setup(self, **kwargs):
        if not switch_to_vessel(self.name):
            return False
        super()._conn_setup(f'r&d: {self.name}')
        return True

    def start(self):
        if not self._conn_setup():
            return
        add_abort_callback(self)

        if not self.plan_flag:
            self.vessel.control.rcs = True
            match_plane_with_target(self.conn, self.spacestation)
            if rendezvous_planner(self.conn, self.spacestation):
                log = f"{self.name}: rendezvous maneuver node created"
                write_log(log)
                self.start_time = self.vessel.control.nodes[0].ut - 300
                self.duration = 216000
                self.tasks.submit_nowait(self)
                self.plan_flag = True
            self.conn.close()
            return

        self.vessel.control.rcs = True
        rendezvous_with_target(self.conn, self.spacestation)
        # TODO: RCS与主引擎控制 多余的航向校正
        # self.spacestation.attitude_adjustment()
        switch_to_vessel(self.name)
        final_approach(self.conn, self.spacestation)
        docking_with_target(self.conn, self.spacestation)

        log = f"{self.name}: docked with {self.spacestation.name}"
        write_log(log)
        self.conn.close()


if __name__ == '__main__':
    from command import ChatMsg
    from task.tasks import Tasks, TaskQueue
    from spacecrafts import Kerbal_Space_Station
    conn = krpc.connect('spacecraft_test')
    station = get_vessel_by_name('KSS')
    KSS = Kerbal_Space_Station()

    # TASK_QUEUE = TaskQueue()
    # msg = ChatMsg('1', '1', '1', '1', 1)
    # tasks = Tasks(msg, 1, 2, TASK_QUEUE)
    # rnd_task = Rendezvous_and_Docking("KSS crew mission 1", KSS, tasks)
    # tasks.submit(rnd_task)
    # TASK_QUEUE.submit(tasks)
    #
    # while True:
    #     tasks = TASK_QUEUE.get()
    #     tasks = tasks.do()
    #     if tasks:
    #         TASK_QUEUE.submit(tasks)
    # match_plane_with_target(conn, KSS)
    # rendezvous_planner(conn, KSS)
    # rendezvous_with_target(conn, KSS)
    final_approach(conn, KSS)
    docking_with_target(conn, KSS)


