from __future__ import annotations
from time import sleep

from .tasks import Task
from .utils import *
from ..astro.orbit import *
from ..astro.maneuver import *
from ..utils import *

if TYPE_CHECKING:
    from .tasks import Tasks
    from ..spacecrafts import Spacecraft
    from krpc.services.spacecenter import Vessel, Node

__all__ = [
    'SimpleMnv',
    'CaptureMnv',
    'ExecuteNode',
]


class SimpleMnv(Task):
    def __init__(self,
                 spacecraft: Spacecraft,
                 tasks: Tasks,
                 mode: str,
                 target: float,
                 orb: Orbit | None = None,
                 start_time: float = -1,
                 duration: float = 300,
                 importance: int = 3,
                 tol: float = 0.2,
                 submit_next: bool = True):
        """进行一次简单的轨道机动规划

        Args:
            name (str): 载具名
            tasks (Tasks): Tasks对象
            mode (str): 机动规划模式(ap, pe, inc)
            target (float): 规划机动的目标
            orb (Orbit, optional): 轨道, 默认规划spacecraft在start_time所处的轨道. Defaults to None.
            start_time (float, optional): 任务执行时. Defaults to -1.
            duration (int, optional): 任务时长. Defaults to 300.
        """
        super().__init__(spacecraft, tasks, start_time, duration, importance, submit_next)
        self.mode = mode
        self.target = target
        self.tol = tol
        if type(orb) is Orbit:
            self.orb = orb
        else:
            self.orb = Orbit.from_krpcv(self.spacecraft.vessel)

    @property
    def description(self):
        return (f'{self.spacecraft.name} 机动节点规划\n'
                f'\t目标{self.mode}: {self.target}\n'
                f'\t预计执行时: {sec_to_date(self.start_time)}')

    @staticmethod
    def _invoke_mnv(mode, orb: Orbit, target: float):
        if mode == 'ap':
            return Maneuver.change_apoapsis(orb, target)
        if mode == 'pe':
            return Maneuver.change_periapsis(orb, target)
        if mode == 'inc':
            orb_t = Orbit.from_coe(orb.attractor, orb.a, orb.e, target,
                                   orb.raan, orb.argp, orb.nu, orb.epoch)
            return Maneuver.match_plane(orb, orb_t, True)
        raise ValueError(f'maneuver mode: {mode}')

    @logging_around
    def start(self):
        if not self._conn_setup():
            return
        time_wrap(self.start_time)
        mnv = self._invoke_mnv(self.mode, self.orb, self.target)
        nodes = mnv.to_krpcv(self.vessel)
        if self.submit_next:
            self._submit_next_task(nodes)
        self.conn.close()

    def _submit_next_task(self, nodes):
        task_list = []
        for n in nodes:
            task_list.append(ExecuteNode.from_node(self.spacecraft, n, self.tasks, tol=self.tol))
        self.tasks.submit_nowait(task_list)

    def _to_dict(self):
        dic = {
            'mode':     self.mode,
            'target':   self.target,
            'tol':      self.tol,
            'orb':      self.orb._to_dict()
            }
        return super()._to_dict() | dic

    @classmethod
    def _from_dict(cls, data, tasks):
        from ..spacecrafts import Spacecraft
        mode = data['mode']
        return cls(
            spacecraft  = Spacecraft.get(data['spacecraft_name']),
            tasks       = tasks,
            mode        = mode,
            target      = data['target'],
            orb         = Orbit._from_dict(data['orb']),
            start_time  = data['start_time'],
            duration    = data['duration'],
            importance  = data['importance'],
            tol         = data['tol'],
            submit_next = data['submit_next']
            )


class CaptureMnv(Task):
    def __init__(self, 
                 spacecraft: Spacecraft, 
                 tasks: Tasks, 
                 orb_t: Orbit,
                 ap_t: float,
                 start_time: float = -1, 
                 duration: float = 300, 
                 importance: int = 9,
                 submit_next: bool = True):
        """捕获机动"""
        super().__init__(spacecraft, tasks, start_time, duration, importance, submit_next)
        self.orb_t = orb_t
        self.ap_t = ap_t

    @property
    def description(self):
        return (f'{self.spacecraft.name} 捕获机动规划\n'
                f'\t{self.orb_t.attractor.name}: {self.ap_t}\n'
                f'\t预计执行时: {sec_to_date(self.start_time)}')

    @logging_around
    def start(self):
        if not self._conn_setup():
            return
        krpc_orb = self.spacecraft.vessel.orbit
        while krpc_orb.body.name != self.orb_t.attractor.name:
            krpc_orb = krpc_orb.next_orbit
            if krpc_orb is None:
                self.conn.close()
                raise RuntimeError(f"{self.spacecraft.name} not in a fly-by orbit of {self.orb_t.attractor.name}")
        orb = Orbit.from_krpcorb(krpc_orb)
        mnv = SimpleMnv(
            self.spacecraft, 
            self.tasks, 
            'ap', 
            self.ap_t, 
            orb = orb,
            importance = self.importance,
            submit_next = self.submit_next,
        )
        self.tasks.submit_nowait(mnv)
        self.conn.close()

    def _to_dict(self):
        dic =  {
            'orb_t':    self.orb_t._to_dict(),
            'ap_t':     self.ap_t
        }
        return super()._to_dict() | dic
    
    @classmethod
    def _from_dict(cls, data, tasks):
        from ..spacecrafts import Spacecraft
        from ..astro.orbit import Orbit
        return cls(
            spacecraft  = Spacecraft.get(data['spacecraft_name']),
            tasks       = tasks,
            orb_t       = Orbit._from_dict(data['orb_t']),
            ap_t        = data['ap_t'],
            start_time  = data['start_time'],
            duration    = data['duration'],
            importance  = data['importance'],
            submit_next = data['submit_next']
        )

        
class ExecuteNode(Task):
    def __init__(self, 
                 spacecraft: Spacecraft, 
                 tasks: Tasks, 
                 burn_time: float,
                 start_time: float = -1, 
                 duration: float = 600, 
                 importance: int = 7,
                 tol: float = 0.2, 
                 submit_next: bool = True):
        """执行最近的一个节点"""
        duration = max(2 * burn_time, duration)
        super().__init__(spacecraft, tasks, start_time, duration, importance, submit_next)
        self.burn_time = burn_time
        self.tol = tol

    @property
    def description(self):
        return (f'{self.spacecraft.name} 轨道机动\n'
                f'\t预计执行时: {sec_to_date(self.start_time)}')
        
    @logging_around
    def start(self):
        if not self._conn_setup():
            return
        if self.burn_time > 3:
            self.spacecraft.part_exts.main_engines = True
        else:
            self.spacecraft.part_exts.main_engines = False
        self.spacecraft.part_exts.rcs = True
        self.executor = self.mj.node_executor
        self.executor.autowarp = True
        self.executor.tolerance = self.tol
        self.executor.execute_one_node()
        while not self.tasks.abort_flag and self.executor.enabled:
            sleep(5)
        self.spacecraft.part_exts.rcs = False
        self.conn.close()

    @classmethod
    def from_node(cls, 
                  spacecraft: Spacecraft, 
                  node: Node, 
                  tasks: Tasks, 
                  importance: int = 7, 
                  tol: float = 0.2, 
                  submit_next: bool = True):
        """执行节点"""
        v = spacecraft.vessel
        # FIXME: 引擎状态不一致
        burn_time = compute_burn_time(
            node.delta_v, 
            spacecraft.part_exts.main_engines, 
            v.mass
        )
        return cls(
            spacecraft  = spacecraft, 
            tasks       = tasks, 
            burn_time   = burn_time,
            start_time  = node.ut - burn_time, 
            duration    = 2 * burn_time, 
            importance  = importance,
            tol         = tol,
            submit_next = submit_next
        )

    def _to_dict(self):
        dic = {
            'burn_time':    self.burn_time,
            'tol':          self.tol
        }
        return super()._to_dict() | dic
    
    @classmethod
    def _from_dict(cls, data, tasks):
        from ..spacecrafts import Spacecraft
        return cls(
            spacecraft  = Spacecraft.get(data['spacecraft_name']),
            tasks       = tasks,
            burn_time   = data['burn_time'],
            start_time  = data['start_time'],
            duration    = data['duration'],
            importance  = data['importance'],
            tol         = data['tol'],
            submit_next = data['submit_next']
        )
