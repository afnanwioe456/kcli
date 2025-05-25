from __future__ import annotations
import numpy as np

from .utils import *
from .astro.orbit import Orbit
from .part_extension import *

if TYPE_CHECKING:
    from .task.tasks import Tasks, Task


class Spacecraft:
    _instances = {}
    
    def __init__(self, name: str):
        self._original_name = name
        self.name, self._tail = self._namer(name)
        Spacecraft._instances[self.name] = self

        self._part_exts: PartExts | None = None
        self._docked_with_name: str | None = None  # 正停泊在的对象, 注意这个记录是单向的

    def __str__(self):
        return self.name

    def _namer(self, name):
        if name not in Spacecraft._instances:
            return name, 1
        tail = 1
        for s in Spacecraft._instances.values():
            s: Spacecraft
            if name == s._original_name and s._tail > tail:
                tail = s._tail
        tail += 1
        return f'{name} #{tail}', tail

    @classmethod
    def get(cls, name) -> Spacecraft:
        return cls._instances.get(name, None)
    
    @classmethod
    def get_or_create(cls, name) -> Spacecraft:
        s = cls._instances.get(name, None)
        if s is None:
            s = cls(name)
        return s
        
    def delete(self):
        Spacecraft._instances.pop(self.name, None)

    @property
    def vessel(self) -> Vessel:
        if self.docked_with:
            v = self.docked_with.vessel
        else:
            v = get_vessel_by_name(self.name)
        return v

    @property
    def part_exts(self):
        if self._part_exts is None:
            self._part_exts = PartExts(self)
        return self._part_exts

    @property
    def docked_with(self):
        if self._docked_with_name is None:
            return None
        return Spacecraft.get(self._docked_with_name)

    def dock_with(self, docking_port_ext: DockingPortExt):
        if not docking_port_ext.part.docking_port.docked_part:
            raise RuntimeError(f'{self} is not docked with target docking port! Call this after docking...')
        spacecraft = docking_port_ext.spacecraft
        self._docked_with_name = spacecraft.name  # 必须先记录_docked_with否则找不到Vessel
        docking_port_ext._dock_with(self)  # 然后更新目标对接口, 否则找不到对接到的DockingPort
        self.part_exts.active_docking_port_ext._dock_with(spacecraft)
        
    @property
    def docking_port_ext_docked_at(self) -> DockingPortExt:
        """返回停靠在的对接口扩展, 未停靠时返回None"""
        if self.docked_with is None:
            return None
        for p in self.docked_with.part_exts.docking_port_exts:
            if p._docked_with_name == self.name:
                return p
        return None

    def undock(self):
        self.part_exts.active_docking_port_ext._undock()
        self.docking_port_ext_docked_at._undock()
        self._docked_with_name = None

    def _to_dict(self):
        return {
            'name':                 self.name,
            '_class_name':          self.__class__.__name__,
            '_original_name':       self._original_name,
            '_tail':                self._tail,
            '_docked_with_name':    self._docked_with_name,
            '_part_exts':           self._part_exts._to_dict(),
        }

    @classmethod
    def _from_dict(cls, data):
        ret                     = cls(data['name'])
        ret._original_name      = data['_original_name']
        ret._tail               = data['_tail']
        ret._docked_with_name   = data['_docked_with_name']
        ret._part_exts          = PartExts._from_dict(data['_part_exts'])
        return ret


class SpacecraftSingleton(Spacecraft):
    _cls_instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._cls_instances:
            cls._cls_instances[cls] = super().__new__(cls)
        return cls._cls_instances[cls]

    def __init__(self, name):
        if '_initialized' in self.__dict__:
            return
        self._initialied = True
        super().__init__(name)


class SpaceStation(SpacecraftSingleton):
    _crew_mission_count = 0
    _supply_mission_count = 0
    _last_mission_end_time = 0

    def supply_mission(self, tasks: Tasks):
        raise NotImplementedError()

    def crew_mission(self, tasks: Tasks):
        raise NotImplementedError()

    @logging_around
    def return_mission(self, docking_port: DockingPortExt, tasks: Tasks) -> list[Task]:
        raise NotImplementedError()

    def _to_dict(self):
        dic = {
            '_crew_mission_count':      self._crew_mission_count,
            '_supply_mission_count':    self._supply_mission_count,
            '_last_mission_end_time':   self._last_mission_end_time,
        }
        return super()._to_dict() | dic

    @classmethod
    def _from_dict(cls, data):
        ret = Spacecraft.get(data['name'])
        if not ret:
            ret = cls(data['name'])
        # 如果已经有初始化的单例则覆盖其属性
        ret._original_name          = data['_original_name']
        ret._tail                   = data['_tail']
        ret._crew_mission_count     = data['_crew_mission_count']
        ret._supply_mission_count   = data['_supply_mission_count']
        ret._last_mission_end_time  = data['_last_mission_end_time']
        ret._docked_with_name       = data['_docked_with_name']
        ret._part_exts              = PartExts._from_dict(data['_part_exts'])
        return ret
        

class KerbalSpaceStation(SpaceStation):
    _deorbit_alt = 50000
    _launch_lead_time = 360
    _min_phase_diff = np.deg2rad(10)
    _max_phase_diff = np.deg2rad(30)
    _time_between_missions = 7776000  # ~90days

    def __init__(self, *args, **kwargs):
        super().__init__('KSS')

    @logging_around
    def crew_mission(self, tasks: Tasks) -> list[Task]: 
        return self._invoke_mission(tasks, 'crew')
    
    @logging_around
    def supply_mission(self, tasks: Tasks) -> list[Task]:
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
        site_p = get_site_position()
        orb = Orbit.from_krpcv(self.vessel)
        if self._last_mission_end_time < orb.epoch:
            self._last_mission_end_time = orb.epoch
        start_time = self._last_mission_end_time
        end_time = self._last_mission_end_time + self._time_between_missions
        self._last_mission_end_time = end_time
        launch_direction = 'SE'
        launch_window = orb.launch_window(
            site_p, 
            direction   = launch_direction, 
            min_phase   = self._min_phase_diff, 
            max_phase   = self._max_phase_diff,
            start_time  = start_time,
            end_time    = end_time
            )
        spacecraft = Spacecraft(f'{self.name} {mission_type} mission {counter}')
        from .task.launch import Soyuz2Launch
        from .task.rendezvous import Rendezvous
        from .task.docking import Docking
        from .task.resource_transfer import ResourceTransfer
        insert_orbit = Orbit.from_coe(
            orb.attractor,
            orb.attractor.r + 240000,
            0.001,
            orb.inc,
            orb.raan,
            orb.argp,
            orb.nu,
            orb.epoch
        )
        launch_task = Soyuz2Launch(
            spacecraft=spacecraft,
            tasks=tasks,
            orbit=insert_orbit,
            # payload_name='_Soyuz_Spacecraft',
            start_time=launch_window - self._launch_lead_time,
            direction=launch_direction
        )
        rdv_task = Rendezvous(spacecraft, self, tasks)
        dock_task = Docking(
            spacecraft, 
            self, 
            tasks
        )
        task_list = [launch_task, rdv_task, dock_task] 
        if mission_type == 'supply':
            resource_task = ResourceTransfer(
               from_spacecraft=spacecraft,
               to_spacecraft=self,
               tasks=tasks,
               trans_all=True 
            )
            task_list.append(resource_task)
        return task_list

    def return_mission(self, docking_port: DockingPortExt, tasks: Tasks) -> list[Task]:
        from .task.maneuver import SimpleMnv
        from .task.docking import Undocking
        from .task.landing import GlideLanding
        s = docking_port.docked_with
        undocking_task = Undocking(s, docking_port, tasks)
        mnv_plan_task = SimpleMnv(s, tasks, 'pe', self._deorbit_alt, importance=0)
        landing_task = GlideLanding(s, tasks)
        return [undocking_task, mnv_plan_task, landing_task]


class GroundStation(SpacecraftSingleton):
    pass
    

KSS = KerbalSpaceStation()

SPACESTATION_DIC: dict[str, SpaceStation] = {
    "kss": KSS
}
