from __future__ import annotations
from enum import Enum
from docopt import docopt

from .utils import *

if TYPE_CHECKING:
    from krpc.services.spacecenter import Part
    from .spacecrafts import Spacecraft


PART_TAG_DOC = f"""
Usage:
    tag [options]
    
Options:
    -d --docking_port       对接口
    -a --active             主动对接口
    -i --resource_in        资源输入
    -o --resource_out       资源输出
    -e --main_engine        主引擎
    --pid <pid>             部件ID
"""


def _part_tag_docopt(part: Part):
    return _part_tag_docopt_at_v(part, vessel=part.vessel)


@temp_switch_to_vessel
def _part_tag_docopt_at_v(part: Part, vessel: Vessel):
    try:
        _args = docopt(PART_TAG_DOC, part.tag.split())
    except SystemExit:
        return
    if _args['--active']:
        _args['--docking_port'] = True
    _args['--title'] = part.title
    return _args


class PartExtType(Enum):
    DOCKING_PORT = '--docking_port'
    RESOURCE_IN = '--resource_in'
    RESOURCE_OUT = '--resource_out'
    MAIN_ENGINE = '--main_engine'

    @classmethod
    def from_tag(cls, tag: str):
        for t in cls:
            if t.value == tag:
                return t
        return None

    def to_class(self):
        if self is PartExtType.DOCKING_PORT:
            return DockingPortExt
        return PartExt

        
class PartExts:
    def __init__(self, spacecraft: Spacecraft):
        self.spacecraft = spacecraft
        self._all: dict[str, PartExt] | None = None

    @property
    def all(self) -> list[PartExt]:
        # 初始化所有PartExt的唯一方法, 其他方法都不会被保存
        if self._all is not None:
            # 如果s已经初始化或恢复所有的扩展部件
            return list(self._all.values())
        # 寻找所有PartExt
        return self._init_all(vessel=self.spacecraft.vessel)

    @temp_switch_to_vessel  # 暂时将画面转到载具上
    def _init_all(self, vessel: Vessel):
        tagged = get_tagged_children_parts(get_root_part(self.spacecraft))
        self._all = PartExt._init_from_krpc_parts(tagged, self.spacecraft)
        return list(self._all.values())

    def update(self):
        self._all = None
        return self.all

    def get_by_type(self, part_type: PartExtType):
        res = []
        for p in self.all:
            if p.part_type == part_type:
                res.append(p)
        return res

    def get_by_id(self, part_id: str) -> PartExt | DockingPortExt | None:
        dummy = self.all
        return self._all.get(part_id, None)

    @property
    def main_engines(self):
        ret = []
        flag = False
        engines = get_parts_in_stage_by_type(
            self.spacecraft.vessel, 
            'engine', 
            self.spacecraft.vessel.control.current_stage)
        for e in engines:
            if PartExtType.MAIN_ENGINE.value in e.tag.split():
                flag = True
                ret.append(e)
        if not flag:
            ret = engines
        return ret
        
    @main_engines.setter
    def main_engines(self, value):
        for e in self.main_engines:
            e.engine.active = value

    @property
    def main_engine_exts(self):
        raise NotImplementedError()

    @property
    def rcs(self):
        return get_parts_in_stage_by_type(
            self.spacecraft.vessel, 
            'rcs', 
            self.spacecraft.vessel.control.current_stage)
    
    @rcs.setter
    def rcs(self, value):
        # TODO: rcs没有激活?
        for e in self.rcs:
            e.rcs.enabled = value

    @property
    def docking_port_exts(self) -> list[DockingPortExt]:
        return self.get_by_type(PartExtType.DOCKING_PORT)

    @property
    def active_docking_port_ext(self):
        dps = self.docking_port_exts
        if not dps:
            return None
        for d in dps:
            if d.args['--active']:
                return d
        return dps[0]

    def get_target_docking_port(self, docking_port_ext: DockingPortExt):
        """返回与docking_port_ext适配的对接口扩展, 优先返回空闲对接口"""
        ret = None
        for d in self.docking_port_exts:
            if d.is_compatible(docking_port_ext):
                if d.is_free():
                    return d
                ret = d
        return ret
            
    @property
    def resource_in_exts(self) -> list[PartExt]:
        return self.get_by_type(PartExtType.RESOURCE_IN)

    @property
    def resource_out_exts(self) -> list[PartExt]:
        return self.get_by_type(PartExtType.RESOURCE_OUT)

    def _to_dict(self):
        if self._all is None:
            return None
        return {
            '_class_name': self.__class__.__name__,
            'spacecraft_name': self.spacecraft.name,
            '_all': {t: self._all.get(t)._to_dict() for t in self._all}
            }

    @classmethod
    def _from_dict(cls, data):
        from .spacecrafts import Spacecraft
        s = Spacecraft.get(data['spacecraft_name'])
        _all: dict = data['_all']
        for k, v in _all.items():
            cls_ = globals().get(v['_class_name'], None)
            if cls_ and issubclass(cls_, PartExt):
                p = cls_._from_dict(v)
                _all[k] = p
            else:
                raise ValueError(f"Unknown or invalid class: {v['_class_name']}")
        # _assign_part_to_exts(s, _all)  不可行, 必须在初始化所有s后
        ret = cls(s)
        ret._all = _all
        return ret


class PartExt:
    def __init__(self,
                 spacecraft: Spacecraft,
                 part_type: PartExtType,
                 part_id: str,
                 args: dict,
                 ):
        # 手动初始化的实例不会被保存
        # 必须通过PartExts.all使用_init_from_krpc_parts实例化
        self.spacecraft = spacecraft
        self.part_type = part_type
        self.part_id = part_id
        self.args = args
        self._part = None

    def __str__(self):
        return f"{self.name} #{self.args['--pid']}"

    @property
    def name(self) -> str:
        return self.args.get('--title')

    @property
    def part(self):
        """krpc Part对象, 注意按照krpc规则调用, 当前载具不一致时无法访问Part属性"""
        if self._part is None:
            self._part = get_part_by_pid(self.spacecraft.vessel, self.part_id)
        return self._part

    @staticmethod
    def _init_from_krpc_parts(parts: Part | list[Part], spacecraft: Spacecraft):
        """返回Part列表对应的PartEx字典(id: PartExt), 如果Part没有id则分配一个
        一个Part对象可能会产生0或多个PartEx对象"""
        res = {}
        if not isinstance(parts, list):
            parts = [parts]
        for p in parts:
            if not p.tag:
                continue
            part_id = get_part_id(p)
            if part_id is None:
                part_id = assign_part_id(p)
            args = _part_tag_docopt(p)
            for t in args:
                # 多个PartEx对象可能指向同一个Part
                if not args[t]:
                    continue
                part_type = PartExtType.from_tag(t)
                if not part_type:
                    continue
                cls_ = part_type.to_class()
                pext = cls_(spacecraft, part_type, part_id, args)
                pext._part = p
                res[part_id] = pext
        return res

    def _to_dict(self):
        return {
            '_class_name': self.__class__.__name__,
            'spacecraft_name': self.spacecraft.name,
            'part_type_value': self.part_type.value,
            'part_id': self.part_id,
            'args': self.args,
            }

    @classmethod
    def _from_dict(cls, data):
        from .spacecrafts import Spacecraft
        s = Spacecraft.get(data['spacecraft_name'])
        part_type = PartExtType.from_tag(data['part_type_value'])
        return cls(
            spacecraft=s,
            part_type = part_type,
            part_id = data['part_id'],
            args = data['args'],
            )


class DockingPortExt(PartExt):
    def __init__(self, 
                 spacecraft: Spacecraft, 
                 part_type: PartExtType,
                 part_id: str,
                 args: dict,
                 ):
        super().__init__(spacecraft, part_type, part_id, args)
        self._docked_with_name: str | None = None
        
    @property
    def docked_with(self) -> Spacecraft:
        from .spacecrafts import Spacecraft
        return Spacecraft.get(self._docked_with_name)

    def is_free(self):
        return self._docked_with_name is None

    def is_docked(self):
        return self._docked_with_name is not None

    def is_compatible(self, docking_port_ext: DockingPortExt):
        return self.name.split()[:-1] == docking_port_ext.name.split()[:-1]

    def _dock_with(self, spacecraft: Spacecraft):
        if self.is_docked():
            raise RuntimeError(f'{self.spacecraft} {self.part.title}({self.part_id}): '
                               f'already docked with {self._docked_with_name}!')
        self._docked_with_name = spacecraft.name

    def _undock(self):
        if not self.is_docked():
            raise RuntimeError(f'{self.spacecraft} {self.part.title}({self.part_id}): '
                               f'not docked with any spacecraft!')
        try:
            self.part.docking_port.undock()
        except RuntimeError:
            pass
        return_s = self.docked_with
        self._docked_with_name = None
        return return_s

    def _to_dict(self):
        dic = {
            '_docked_with_name': self._docked_with_name
        }
        return dic | super()._to_dict()

    @classmethod
    def _from_dict(cls, data):
        from .spacecrafts import Spacecraft
        s = Spacecraft.get(data['spacecraft_name'])
        part_type = PartExtType.from_tag(data['part_type_value'])
        ret = cls(
            spacecraft=s,
            part_type = part_type,
            part_id = data['part_id'],
            args = data['args'],
            )
        ret._docked_with_name = data['_docked_with_name']
        return ret
    
