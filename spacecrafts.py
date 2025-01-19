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
        v = get_vessel_by_name(self.name)
        if not v:
            raise RuntimeError(f'{self.name} does not exist')
        return v


class Spacecraft(SpacecraftBase):
    def __init__(self, name):
        super().__init__(name)

    @property
    def main_engine(self):
        ret = []
        flag = False
        engines = get_parts_in_stage_by_type(self.vessel, 'engine', self.vessel.control.current_stage)
        for e in engines:
            if 'main' in e.tag.split():
                flag = True
                ret.append(e)
        if not flag:
            ret = engines
        return ret
        
    @main_engine.setter
    def main_engine(self, value):
        if not isinstance(value, bool):
            raise ValueError(f"Value must be bool, got {type(value).__name__}")
        for e in self.main_engine:
            e.engine.active = value

    @property
    def rcs(self):
        return get_parts_in_stage_by_type(self.vessel, 'rcs', self.vessel.control.current_stage)
    
    @rcs.setter
    def rcs(self, value):
        if not isinstance(value, bool):
            raise ValueError(f"Value must be bool, got {type(value).__name__}")
        for e in self.rcs:
            e.rcs.active = value


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
        dps = self.crew_docking_ports if docking_port_type is DockingPortType.crew else self.supply_docking_ports
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
    _insert_orbit = (240000, 240000)
    _launch_lead_time = 360
    _min_phase_diff = 10
    _max_phase_diff = 40

    def __init__(self):
        super().__init__('KSS')
        self._docking_ports = [
            DockingPortStatus(self, '1', DockingPortType.crew),
            DockingPortStatus(self, '2', DockingPortType.supply),
        ]

    @logging_around
    def crew_mission(self, tasks: Tasks) -> list[Task]: 
        return self._invoke_mission(tasks, DockingPortType.crew)
    
    @logging_around
    def supply_mission(self, tasks: Tasks):
        return self._invoke_mission(tasks, DockingPortType.supply)

    def _invoke_mission(self, tasks: Tasks, dp_type):
        # TODO: 重复创建任务冲突问题
        self._crew_mission_count += 1
        site_p = get_launch_site_position()
        orb = Orbit.from_krpcv(self.vessel)
        launch_window = Orbit.launch_window(orb, 
                                            site_p, 
                                            'SE', 
                                            min_phase=self._min_phase_diff, 
                                            max_phase=self._max_phase_diff)
        launch_window = launch_window.to_value(u.s)
        inc = -np.degrees(self.vessel.orbit.inclination)

        spacecraft = Spacecraft(f'{self.name} crew mission {self._crew_mission_count}')
        from .task.launch import Soyuz2Launch
        from .task.rendezvous import Rendezvous
        from .task.docking import Docking
        launch_task = Soyuz2Launch(tasks=tasks,
                                   spacecraft=spacecraft,
                                   # payload_name='_Soyuz_Spacecraft',
                                   ap_altitude=self._insert_orbit[0],
                                   pe_altitude=self._insert_orbit[1],
                                   inclination=inc,
                                   start_time=launch_window - self._launch_lead_time)
        rdv_task = Rendezvous(spacecraft, self, tasks)
        dock_task = Docking(spacecraft, 
                            self, 
                            self._get_target_docking_port(dp_type), 
                            tasks)
        return [launch_task, rdv_task, dock_task]

    def _crew_return_mission(self, docking_port: DockingPortStatus, tasks: Tasks) -> list[Task]:
        deorbit_alt = 50000
        from .task.maneuver import SimpleMnv
        from .task.docking import Undocking
        undocking_task = Undocking(self, docking_port, tasks)
        mnv_plan_task = SimpleMnv(docking_port.docked_with, tasks, 'pe', deorbit_alt * u.m, importance=0)
        # TODO: 回收
        return [undocking_task, mnv_plan_task]

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
        raise RuntimeError(f'{self.spacecraft} docking port [{self.num}] does not exist.')
    
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
