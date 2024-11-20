from __future__ import annotations

from time import sleep

from task.tasks import Task
from utils import *

if TYPE_CHECKING:
    from task.tasks import Tasks


class ReleasePayload(Task):
    def __init__(self,
                 name,  # 载荷名称
                 tasks: Tasks,
                 count=1,  # 待分离的载荷数
                 wait_time=10,
                 start_time=-1,
                 duration=180,
                 ):
        super().__init__(name, tasks, start_time, duration)
        self.importance = 10
        self.count = count
        self.wait_time = wait_time
        self.original_name = get_original_name(self.name)

        self.conn = None
        self.vessel = None

    @property
    def description(self):
        return (f'{self.name} 释放载荷: {self.count}\n'
                f'\t预计执行时: {sec_to_date(self.start_time)}')

    def _conn_setup(self, **kwargs):
        if not switch_to_vessel(self.name):
            return False
        super()._conn_setup(f'{self.name} payload release')
        return True

    @staticmethod
    def _deploy(vessel: Vessel,
                wait=5,
                antenna=True,
                solar_panel=True,
                ):
        sleep(wait)
        if antenna:
            deploy_antenna(vessel)
        if solar_panel:
            deploy_solar_panels(vessel)

    def start(self):
        setup_flag = self._conn_setup()
        if not setup_flag or self.conn is None or self.sc is None or self.vessel is None:
            return
        self.sc.quicksave()
        sleep(5)

        self.vessel.name = vessel_namer(f'{self.name}_tmp')
        # 分级，释放第一个载荷会自动切换到载荷上
        self._activate_next_stage()
        print(f'释放载荷：{self.vessel.name}')
        self.count -= 1
        adapter_name = self.vessel.name
        # 将active_vessel切换到释放的载具上并命名
        self.vessel = self.sc.active_vessel
        self.vessel.name = self.name
        self._deploy(self.vessel)

        # 尝试切换到适配器上
        if switch_to_vessel(adapter_name):
            self.vessel = self.sc.active_vessel
        elif self.conn is not None:
            self.conn.close()
            return
        else:
            return

        # 如果还有载荷
        while self.count > 0:
            past_vessels = self.sc.vessels
            # 分级，注意ksp不会自动切换到载荷上
            self._activate_next_stage()
            current_vessels = self.sc.vessels
            new_payloads = get_new_vessels(past_vessels, current_vessels)
            for p in new_payloads:
                p.name = vessel_namer(self.original_name)
                self._deploy(p, 3)
                print(f'释放载荷：{p.name}')
            sleep(5)
            self.count -= 1

        self.vessel.type = self.sc.VesselType.debris # type: ignore
        self.vessel.name += ' Debris'
        print('payload release complete.')
        self.conn.close()


def deploy_solar_panels(vessel: Vessel):
    solar_panels = vessel.parts.solar_panels
    for sp in solar_panels:
        if sp.deployable:
            sp.deployed = True


def deploy_antenna(vessel: Vessel):
    antennas = vessel.parts.antennas
    for a in antennas:
        if a.deployable:
            a.deployed = True


if __name__ == '__main__':
    from task.tasks import TaskQueue
    TASK_QUEUE = TaskQueue()
    from threading import Event
    NEW_TASK_EVENT = Event()
    NEW_TASK_EVENT.set()
    t = ''
    release = ReleasePayload("Ariane_5_ECA_Relay", t, 2) # type: ignore
    release.start()
