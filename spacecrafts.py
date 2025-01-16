from __future__ import annotations
from functools import cached_property
from time import sleep
import numpy as np
from astropy import units as u

from .utils import *
from .astro.orbit import Orbit

if TYPE_CHECKING:
    from .task.tasks import Tasks, Task


class SpacecraftBase:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    @property
    def vessel(self):
        return get_vessel_by_name(self.name)


class SpaceStation(SpacecraftBase):
    _crew_mission_count = 0
    _supply_mission_count = 0

    def __init__(self, name):
        super().__init__(name)
        self._docking_ports: list[DockingPortStatus] = []

    @cached_property
    def supply_docking_ports(self):
        dps: list[DockingPortStatus] = []
        for p in self._docking_ports:
            if p.is_supply():
                dps.append(p)
        return dps

    @cached_property
    def crew_docking_ports(self):
        dps: list[DockingPortStatus] = []
        for p in self._docking_ports:
            if p.is_crew():
                dps.append(p)
        return dps

    def _get_target_docking_port(self, docking_port_type):
        dps = self.supply_docking_ports if docking_port_type is DockingPortType.supply else self.crew_docking_ports
        for p in dps:
            if p.is_free():
                return p
        return dps[0]

    def supply_mission(self, tasks: Tasks):
        return

    def _supply_return_mission(self, docking_port: DockingPortStatus):
        return

    def crew_mission(self, tasks: Tasks):
        return

    def _crew_return_mission(self, docking_port: DockingPortStatus):
        return

    @logging_around
    def return_mission(self, docking_port: DockingPortStatus, tasks: Tasks) -> list[Task]:
        if docking_port.is_crew():
            return self._crew_return_mission(docking_port, tasks)
        if docking_port.is_supply():
            return self._supply_return_mission(docking_port, tasks)
        raise NotImplementedError()


class KerbalSpaceStation(SpaceStation):
    insert_orbit = (240000, 240000)
    launch_lead_time = 360
    launch_phase_diff = 40

    def __init__(self):
        super().__init__('KSS')
        self._docking_ports = [
            DockingPortStatus(self, '1', DockingPortType.crew),
            DockingPortStatus(self, '2', DockingPortType.supply),
        ]

    @logging_around
    def crew_mission(self, tasks: Tasks) -> list[Task]: 
        self._crew_mission_count += 1
        site_p = get_launch_site_position()
        orb = Orbit.from_krpcv(self.vessel)
        launch_window = Orbit.launch_window(orb, site_p, 'SE', min_phase=self.launch_phase_diff)
        launch_window = launch_window.to_value(u.s)
        inc = -np.degrees(self.vessel.orbit.inclination)

        spacecraft_name = f'{self.name} crew mission {self._crew_mission_count}'
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
        dock_task = Docking(spacecraft_name, 
                            self, 
                            self._get_target_docking_port(DockingPortType.crew), 
                            tasks)
        return [launch_task, rdv_task, dock_task]
    
    def _crew_return_mission(self, docking_port: DockingPortStatus, tasks: Tasks) -> list[Task]:
        deorbit_alt = 50000

        from .task.maneuver import SimpleMnv
        from .task.docking import Undocking
        undocking_task = Undocking(self, docking_port, tasks)
        mnv_plan_task = SimpleMnv(docking_port.docked_with.name, tasks, 'pe', deorbit_alt * u.m, importance=0)
        # TODO: 回收
        return [undocking_task, mnv_plan_task]

    def supply_mission(self, tasks: Tasks):
        # TODO:
        return self.crew_mission(tasks)

    def _supply_return_mission(self, docking_port, tasks) -> list[Task]:
        return self._crew_return_mission(docking_port, tasks)
        

class DockingPortStatus:
    def __init__(self, 
                 spacecraft: SpaceStation, 
                 num: str, 
                 type: int):
        self.spacecraft = spacecraft
        self.num = num
        self.docked_with: SpacecraftBase | None = None
        self._type = type
        self._scheduled_count = 0

    def is_free(self):
        return self.docked_with is None and self._scheduled_count == 0

    def is_scheduled(self):
        return self._scheduled_count > 0
    
    def is_docked(self):
        return self.docked_with is not None

    def is_crew(self):
        return self._type is DockingPortType.crew
    
    def is_supply(self):
        return self._type is DockingPortType.supply

    @property
    def part(self):
        for p in self.spacecraft.vessel.parts.docking_ports:
            if self.num in p.part.tag.split(' '):
                return p
    
    @staticmethod
    def from_krpc(part):
        raise NotImplementedError()

    def schedule(self):
        self._scheduled_count += 1

    def unschedule(self):
        if not self.is_scheduled():
            raise RuntimeError(f'{self.spacecraft} docking port [{self.num}]: not scheduled!')
        self._scheduled_count -= 1

    def dock_with(self, spacecraft):
        if self.is_docked():
            raise RuntimeError(f'{self.spacecraft} docking port [{self.num}]: already docked with {self.docked_with}!')
        self.docked_with = spacecraft

    def undock(self):
        if not self.is_docked():
            raise RuntimeError(f'{self.spacecraft} docking port [{self.num}]: not docked with any spacecraft!')
        v = self.part.undock()
        if v.name != self.docked_with.name:
            raise RuntimeError(f'Inconsistent spacecrafts: {v.name}, {self.docked_with.name}')
        return_sc = self.docked_with
        self.docked_with = None
        return return_sc


class DockingPortType:
    supply = 0
    crew = 1


KSS = KerbalSpaceStation()

SPACESTATION_DIC: dict[str, SpaceStation] = {
    "kss": KSS
}
