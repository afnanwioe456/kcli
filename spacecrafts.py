from __future__ import annotations
from functools import cached_property
from typing import final
import threading
import numpy as np
from astropy import units as u

from .utils import *
from .astro.orbit import Orbit

if TYPE_CHECKING:
    from .task.tasks import Tasks, Task


class SpacecraftBase:
    _spacecraft_dic = {}
    
    def __init__(self, name: str):
        self._original_name = name
        self.name, self._tail = self._namer(name)
        self._spacecraft_dic[self.name] = self

    def __str__(self):
        return self.name

    def _namer(self, name):
        if name not in self._spacecraft_dic.keys():
            return name, 1
        tail = 1
        for s in self._spacecraft_dic.values():
            if name == s._original_name and s._tail > tail:
                tail = s._tail
        tail += 1
        return f'{name} #{tail}', tail

    @final
    @classmethod
    def get(cls, name) -> SpacecraftBase:
        return cls._spacecraft_dic.get(name, None)
        
    def remove(self):
        self._spacecraft_dic.pop(self.name, None)

    @property
    def vessel(self):
        v = get_vessel_by_name(self.name)
        if not v:
            raise RuntimeError(f'{self.name} does not exist')
        return v

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
            e.rcs.enabled = value

    def _to_dict(self):
        return {
            'type': self.__class__.__name__,
            '_original_name': self._original_name,
            '_tail': self._tail,
            'name': self.name,
        }

    @classmethod
    def _from_dict(cls, data):
        ret = cls(data['name'])
        ret._original_name = data['_original_name']
        ret._tail = data['_tail']
        return ret

    @classmethod
    def dump_all(cls):
        data = [s._to_dict() for s in cls._spacecraft_dic.values()]
        return data

    @staticmethod
    def load_all(data):
        for item in data:
            class_name = item['type']
            cls = globals().get(class_name, None)
            if cls and issubclass(cls, SpacecraftBase):
                cls._from_dict(item)
            else:
                raise ValueError(f'Unknown or invalid class type: {class_name}')


class Spacecraft(SpacecraftBase):
    def __init__(self, name: str):
        super().__init__(name)


class SpaceStation(SpacecraftBase):
    _crew_mission_count = 0
    _supply_mission_count = 0
    _last_mission_end_time = 0 * u.s
    _instances = {}
    _new_lock = threading.Lock()

    def __new__(cls):
        if cls not in cls._instances:
            with cls._new_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]

    def __init__(self, name: str):
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

    def get_docking_port(self, num: str):
        num = str(num)
        for dp in self._docking_ports:
            if dp.num == num:
                return dp

    def _get_target_docking_port(self, docking_port_type: str):
        if docking_port_type == 'crew':
            dps = self.crew_docking_ports
        elif docking_port_type == 'supply':
            dps = self.supply_docking_ports
        else:
            raise ValueError(docking_port_type)
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

    def _to_dict(self):
        dic = {
        '_crew_mission_count': self._crew_mission_count,
        '_supply_mission_count': self._supply_mission_count,
        '_docking_ports': [dp._to_dict() for dp in self._docking_ports],
        '_last_mission_end_time': self._last_mission_end_time.to_value(u.s),
        }
        return super()._to_dict() | dic

    @classmethod
    def _from_dict(cls, data):
        ret = cls(data['name'])
        ret._original_name = data['_original_name']
        ret._tail = data['_tail']
        ret._crew_mission_count = data['_crew_mission_count']
        ret._supply_mission_count = data['_supply_mission_count']
        ret._last_mission_end_time = data['_last_mission_end_time'] * u.s
        ret._docking_ports = [DockingPortStatus._from_dict(d) for d in data['_docking_ports']]
        return ret
        

class KerbalSpaceStation(SpaceStation):
    _insert_orbit = (240000, 240000)
    _deorbit_alt = 50000
    _launch_lead_time = 360
    _min_phase_diff = 10 * u.deg
    _max_phase_diff = 40 * u.deg
    _time_between_missions = 90 * u.d

    def __init__(self, *args, **kwargs):
        super().__init__('KSS')
        self._docking_ports = [
            DockingPortStatus(self, '1', DockingPortType.crew),
            DockingPortStatus(self, '2', DockingPortType.supply),
        ]

    @logging_around
    def crew_mission(self, tasks: Tasks) -> list[Task]: 
        return self._invoke_mission(tasks, 'crew')
    
    @logging_around
    def supply_mission(self, tasks: Tasks):
        return self._invoke_mission(tasks, 'supply')

    def _invoke_mission(self, tasks: Tasks, mission_type: str):
        if mission_type == 'crew':
            self._crew_mission_count += 1
            counter = self._crew_mission_count
        elif mission_type == 'supply':
            self._supply_mission_count += 1
            counter = self._supply_mission_count
        else:
            raise ValueError(mission_type)
        site_p = get_launch_site_position()
        orb = Orbit.from_krpcv(self.vessel)
        if self._last_mission_end_time < orb.epoch:
            self._last_mission_end_time = orb.epoch
        end_time = self._last_mission_end_time + self._time_between_missions
        launch_window = Orbit.launch_window(orb, 
                                            site_p, 
                                            direction='SE', 
                                            min_phase=self._min_phase_diff, 
                                            max_phase=self._max_phase_diff,
                                            start_time=self._last_mission_end_time,
                                            end_time=end_time)
        self._last_mission_end_time = end_time
        launch_window = launch_window.to_value(u.s)
        inc = -np.degrees(self.vessel.orbit.inclination)

        spacecraft = Spacecraft(f'{self.name} crew mission {counter}')
        from .task.launch import Soyuz2Launch
        from .task.rendezvous import Rendezvous
        from .task.docking import Docking
        launch_task = Soyuz2Launch(spacecraft=spacecraft,
                                   tasks=tasks,
                                   # payload_name='_Soyuz_Spacecraft',
                                   ap_altitude=self._insert_orbit[0],
                                   pe_altitude=self._insert_orbit[1],
                                   inclination=inc,
                                   start_time=launch_window - self._launch_lead_time)
        rdv_task = Rendezvous(spacecraft, self, tasks)
        dock_task = Docking(spacecraft, 
                            self, 
                            self._get_target_docking_port(mission_type), 
                            tasks)
        return [launch_task, rdv_task, dock_task]

    def _crew_return_mission(self, docking_port: DockingPortStatus, tasks: Tasks) -> list[Task]:
        from .task.maneuver import SimpleMnv
        from .task.docking import Undocking
        from .task.landing import GlideLanding
        s = docking_port.docked_with
        undocking_task = Undocking(self, docking_port, tasks)
        mnv_plan_task = SimpleMnv(s, tasks, 'pe', self._deorbit_alt * u.m, importance=0)
        landing_task = GlideLanding(s, tasks)
        return [undocking_task, mnv_plan_task, landing_task]

    def _supply_return_mission(self, docking_port: DockingPortStatus, tasks: Tasks) -> list[Task]:
        from .task.maneuver import SimpleMnv
        from .task.docking import Undocking
        from .task.landing import ControlledReentry
        s = docking_port.docked_with
        undocking_task = Undocking(self, docking_port, tasks)
        mnv_plan_task = SimpleMnv(s, tasks, 'pe', self._deorbit_alt * u.m, importance=0)
        reentry_task = ControlledReentry(s, tasks)
        return [undocking_task, mnv_plan_task, reentry_task]
        

class DockingPortStatus:
    def __init__(self, 
                 spacecraft: SpacecraftBase, 
                 num: str, 
                 type: int):
        self._spacecraft_name = spacecraft.name
        self.num = num
        self._docked_with: str | None = None
        self._type = type
        
    @property
    def spacecraft(self) -> SpacecraftBase:
        return SpacecraftBase.get(self._spacecraft_name)

    @property
    def docked_with(self) -> SpacecraftBase:
        return SpacecraftBase.get(self._docked_with)

    @docked_with.setter
    def docked_with(self, spacecraft: SpacecraftBase | None):
        if spacecraft is None:
            self._docked_with = None
            return
        self._docked_with = spacecraft.name

    def is_free(self):
        return self._docked_with is None

    def is_docked(self):
        return self._docked_with is not None

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

    def _to_dict(self):
        return {
        '_spacecraft': self._spacecraft_name,
        'num': self.num,
        '_docked_with': self._docked_with,
        '_type': self._type,
        }

    @classmethod
    def _from_dict(cls, data):
        ret = cls(SpacecraftBase.get(data['_spacecraft']), data['num'], data['_type'])
        ret._docked_with = data['_docked_with']
        return ret


class DockingPortType:
    supply = 0
    crew = 1


KSS = KerbalSpaceStation()

SPACESTATION_DIC: dict[str, SpaceStation] = {
    "kss": KSS
}
