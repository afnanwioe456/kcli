from __future__ import annotations

import numpy as np

from utils import *
import astrodynamics as astro

if TYPE_CHECKING:
    from task.tasks import Task, Tasks


class Spacestation:
    name = 'undefined space station'
    crew_mission_count = 0
    supply_mission_count = 0

    def __init__(self):
        self.vessel = get_vessel_by_name(self.name)
        self.control = self.vessel.control
        self.autopilot = self.vessel.auto_pilot

    def resource_calculater(self):
        pass

    @property
    def supply_docking_ports(self):
        available_ports = []
        for p in self.vessel.parts.docking_ports:
            if 'supply' in p.part.tag and not p.docked_part:
                available_ports.append(p)
        return available_ports

    @property
    def crew_docking_ports(self):
        available_ports = []
        for p in self.vessel.parts.docking_ports:
            if 'crew' in p.part.tag and not p.docked_part:
                available_ports.append(p)
        return available_ports

    def attitude_adjustment(self):
        switch_to_vessel(self.name)
        self.control.rcs = True
        self.autopilot.sas = True
        self.autopilot.sas_mode = UTIL_CONN.space_center.SASMode.prograde
        self.autopilot.target_roll = 0
        self.autopilot.engage()
        self.autopilot.wait()
        self.control.rcs = False

    def supply_mission(self, tasks: Tasks) -> list[Task] | None:
        """返回补给任务的任务列表[Task]"""
        return

    def crew_mission(self, tasks: Tasks) -> list[Task] | None:
        """返回成员任务的任务列表[Task]"""
        return


class Kerbal_Space_Station(Spacestation):
    name = 'KSS'
    insert_orbit = (240000, 240000)
    launch_lead_time = 360
    launch_phase_angle = 20

    def __init__(self):
        super().__init__()

    def crew_mission(self, tasks: Tasks):
        # TODO: 撤离空间站
        self.crew_mission_count += 1
        launch_windows = astro.orbit_launch_window(self.vessel, direction='SE')
        best_launch_window = launch_windows[1][0]
        inc = -np.degrees(self.vessel.orbit.inclination)

        spacecraft_name = f'{self.name} crew mission {self.crew_mission_count}'
        from launch import Soyuz2Launch
        from rendezvous_and_docking import Rendezvous_and_Docking
        launch_task = Soyuz2Launch(tasks,
                                   name=spacecraft_name,
                                   # payload_name='_Soyuz_Spacecraft',
                                   ap_altitude=self.insert_orbit[0],
                                   pe_altitude=self.insert_orbit[1],
                                   inclination=inc,
                                   start_time=best_launch_window - self.launch_lead_time)
        rnd_task = Rendezvous_and_Docking(spacecraft_name, self, tasks)

        return [launch_task, rnd_task]


KSS = Kerbal_Space_Station()

SPACESTATION_NAME_DIC: [str, str] = {
    "kss": "近地空间站"
}

SPACESTATION_DIC: [str, Spacestation] = {
    "kss": KSS
}


if __name__ == '__main__':
    from command import ChatMsg
    from task.tasks import TaskQueue
    from task.tasks import Tasks

    TASK_QUEUE = TaskQueue()

    msg = ChatMsg('1', '1', '1', '1', 1)
    tasks = Tasks(msg, 1, 2, TASK_QUEUE)
    tasks.submit(KSS.crew_mission(tasks))
    TASK_QUEUE.put(tasks)
    while True:
        tasks = TASK_QUEUE.get()
        tasks = tasks.do()
        if tasks:
            TASK_QUEUE.submit(tasks)
