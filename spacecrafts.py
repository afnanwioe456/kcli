from __future__ import annotations
import numpy as np
from astropy import units as u

from .utils import *
from .astro.orbit import Orbit

if TYPE_CHECKING:
    from .task.tasks import Task, Tasks


class Spacestation:
    def __init__(self, name):
        self.name = name
        self.crew_mission_count = 0
        self.supply_mission_count = 0
        self.vessel = get_vessel_by_name(self.name)
        self.control = self.vessel.control
        self.autopilot = self.vessel.auto_pilot

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

    @logging_around
    def supply_mission(self, tasks: Tasks) -> list[Task] | None:
        """返回补给任务的任务列表[Task]"""
        return

    @logging_around
    def crew_mission(self, tasks: Tasks) -> list[Task] | None:
        """返回成员任务的任务列表[Task]"""
        return


class Kerbal_Space_Station(Spacestation):
    insert_orbit = (240000, 240000)
    launch_lead_time = 360
    launch_phase_diff = 50

    def __init__(self):
        super().__init__('KSS')

    @logging_around
    def crew_mission(self, tasks: Tasks): 
        # TODO: 撤离空间站
        self.crew_mission_count += 1
        site_p = get_launch_site_position()
        orb = Orbit.from_krpcv(self.vessel)
        launch_window = Orbit.launch_window(orb, site_p, 'SE', min_phase=self.launch_phase_diff)
        launch_window = launch_window.to_value(u.s)
        inc = -np.degrees(self.vessel.orbit.inclination)

        spacecraft_name = f'{self.name} crew mission {self.crew_mission_count}'
        from .task.launch import Soyuz2Launch
        from .task.rendezvous import Rendezvous
        from .task.docking import Docking
        launch_task = Soyuz2Launch(tasks,
                                   name=spacecraft_name,
                                   # payload_name='_Soyuz_Spacecraft',
                                   ap_altitude=self.insert_orbit[0],
                                   pe_altitude=self.insert_orbit[1],
                                   inclination=inc,
                                   start_time=launch_window - self.launch_lead_time)
        rdv_task = Rendezvous(spacecraft_name, self, tasks)
        dock_task = Docking(spacecraft_name, self, tasks)
        tasks.submit_nowait([launch_task, rdv_task, dock_task])
        return tasks


KSS = Kerbal_Space_Station()

SPACESTATION_NAME_DIC: dict[str, str] = {
    "kss": "近地空间站"
}

SPACESTATION_DIC: dict[str, Spacestation] = {
    "kss": KSS
}
