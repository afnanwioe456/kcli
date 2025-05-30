from __future__ import annotations
from time import sleep
from krpc.services.spacecenter import ResourceTransfer as RT

from .tasks import *
from .utils import *
from ..part_extension import *
from ..utils import *


__all__ = [
    'ResourceTransfer',
]


def _transfer_resource(out_parts: list[Part],
                       in_parts: list[Part],
                       resource: str,
                       amount: float,
                       rt: RT):
    A = [a.resources.amount(resource) for a in out_parts]
    B = [b.resources.amount(resource) for b in in_parts]
    cap = [b.resources.max(resource) for b in in_parts]
    matrix = waterfill_matrix(A, B, cap, amount)
    print(matrix)
    for i in range(len(out_parts)):
        op = out_parts[i]
        for j in range(len(in_parts)):
            ip = in_parts[j]
            rt.start(op, ip, resource, matrix[i, j])


class ResourceTransfer(Task):
    def __init__(self, 
                 from_spacecraft: Spacecraft, 
                 to_spacecraft: Spacecraft,
                 tasks: Tasks, 
                 resources: dict[str, float] | None = None,
                 trans_all: bool = False,
                 start_time: float = -1, 
                 duration: float = 600, 
                 importance: int = 3, 
                 submit_next: bool = True):
        super().__init__(from_spacecraft, tasks, start_time, duration, importance, submit_next)
        self.from_spacecraft = from_spacecraft
        self.to_spacecraft = to_spacecraft
        if resources is None:
            resources = {}
        self.resources = resources
        self.trans_all = trans_all

    @property
    def description(self):
        return (f'{self.from_spacecraft.name} -> 资源转移 -> {self.spacecraft.name}'
                f'\t预计执行时: {sec_to_date(self.start_time)}')

    @logging_around
    def start(self):
        if not self._conn_setup():
            return
        out_exts = self.from_spacecraft.part_exts.resource_out_exts
        in_exts = self.to_spacecraft.part_exts.resource_in_exts
        if self.trans_all:
            for ext in out_exts:
                for r in ext.part.resources.all:
                    self.resources[r.name] = np.inf
        # TODO: 在classmethod修复之前暂时不要访问rt的任何属性
        rt = RT(self.conn, -1)
        for r, a in self.resources.items():
            out_parts = []
            for ext in out_exts:
                if ext.part.resources.has_resource(r):
                    out_parts.append(ext.part)
            in_parts = []
            for ext in in_exts:
                if ext.part.resources.has_resource(r):
                    in_parts.append(ext.part)
            _transfer_resource(out_parts, in_parts, r, a, rt)            
        sleep(10)

    def _to_dict(self):
        dic = {
            'from_spacecraft_name': self.from_spacecraft.name,
            'to_spacecraft_name':   self.to_spacecraft.name,
            'resources':            self.resources,
            'trans_all':            self.trans_all,
        }
        return super()._to_dict() | dic

    @classmethod
    def _from_dict(cls, data, tasks):
        from ..spacecrafts import Spacecraft
        _from = Spacecraft.get(data['from_spacecraft_name'])
        _to = Spacecraft.get(data['to_spacecraft_name'])
        return cls(
            from_spacecraft = _from,
            to_spacecraft   = _to,
            tasks           = tasks,
            resources       = data['resources'],
            trans_all       = data['trans_all'],
            start_time      = data['start_time'],
            duration        = data['duration'],
            importance      = data['importance'],
            submit_next     = data['submit_next'],
        )