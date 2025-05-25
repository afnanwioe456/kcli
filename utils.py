from __future__ import annotations
from typing import TYPE_CHECKING
from dateutil import parser, tz
from datetime import datetime, timedelta
import random
import string
import logging

import krpc

if TYPE_CHECKING:
    from krpc.services.spacecenter import Vessel, Part
    from .spacecrafts import Spacecraft
    from .part_extension import PartExt

_DEBUG_MODE = False
KSP_EPOCH_TIME = -599616000
LAUNCH_SITES_COORDS = {
    'wenchang': (19.613726150307052, 110.9553275138089)
}

### LOGGER ###

def setup_logger():
    logger = logging.getLogger("kcli")
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("./.live/log.txt", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    debug_file_handler = logging.FileHandler("./.live/debug.log", encoding="utf-8")
    debug_file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    debug_file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(debug_file_handler)
    logger.addHandler(console_handler)

    return logger

LOGGER = setup_logger()

def logging_around(func):
    def wrapper(*args, **kwargs):
        log_flag = False
        if len(args) > 0 and isinstance(args[0], object):
            instance = args[0]
            class_name = instance.__class__.__name__
            func_name = func.__name__
            attrs = dir(instance)
            task_name = instance.name if 'name' in attrs else 'Instance'
            LOGGER.debug(f'{task_name} entering {class_name}.{func_name}')
            log_flag = True
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            if log_flag:
                LOGGER.exception(f'{task_name} raised an execption in {class_name}.{func_name}')
            raise e
        if log_flag:
            LOGGER.debug(f'{task_name} exiting {class_name}.{func_name}')
        return result
    return wrapper
            
### GLOBAL CONNECTION ###

UTIL_CONN = None if _DEBUG_MODE else krpc.connect('kcli')

### IN-GAME TIME CONTROL ###

def get_ut() -> float:
    if _DEBUG_MODE:
        return 0
    return UTIL_CONN.space_center.ut


def time_wrap(start_time):
    ut = UTIL_CONN.space_center.ut
    if ut < start_time:
        LOGGER.debug(f'time wrapping to {sec_to_date(start_time)} ...')
        UTIL_CONN.space_center.warp_to(start_time, max_rails_rate=1e8)


def date_to_sec(input_date: str) -> float | None:
    """
    日期转换为自1951-01-01 00:00:00的时间, 无法解析时返回None
    """
    try:
        # 尝试使用dateutil.parser解析输入
        parsed_date = parser.parse(input_date).replace(tzinfo=tz.tzutc())
        # 如果没有输入日期
        if parsed_date.date() == parser.parse('00:00:00').date():
            default_date = str(sec_to_date(get_ut()).date())
            parsed_date = parser.parse(f'{default_date} {input_date}').replace(tzinfo=tz.tzutc())
        epoch_date = parser.parse("1970-01-01 00:00:00 UTC")
        time_difference = parsed_date - epoch_date
        seconds_since_epoch = int(time_difference.total_seconds())
        seconds_since_ksp_epoch = seconds_since_epoch - KSP_EPOCH_TIME
    except ValueError:
        LOGGER.warning("无法解析输入的时间格式，请确保输入的格式为支持的格式。")
        return

    if seconds_since_ksp_epoch is not None:
        return seconds_since_ksp_epoch


def sec_to_date(seconds: float) -> datetime:
    """
    自1951-01-01 00:00:00的时间转换为datetime对象
    """
    epoch = datetime(1951, 1, 1, 0, 0, 0)
    time_difference = timedelta(seconds=seconds)
    result_datetime = epoch + time_difference
    return result_datetime

### LAUNCH_SITE ###

def get_site_position(name: str = 'wenchang'):
    """发射场或载具的位置矢量"""
    pos = LAUNCH_SITES_COORDS.get(name, None)
    if pos is None:
        v = get_vessel_by_name(name)
        if v is not None:  
            # 如果传入的载具已经在任务中, 则将它作为发射位置
            site_p = v.position(v.orbit.body.non_rotating_reference_frame)
            site_p = (site_p[0], site_p[2], site_p[1])
            return site_p
        else:
            pos = LAUNCH_SITES_COORDS.get('wenchang')
    la, lo = pos
    body = UTIL_CONN.space_center.bodies['Earth']
    site_p = body.surface_position(la, lo, body.non_rotating_reference_frame)
    site_p = (site_p[0], site_p[2], site_p[1])
    return site_p

def dummy_roll_out():
    """用于将游戏场景切换回"""
    sc = UTIL_CONN.space_center
    sc.launch_vessel('VAB', 'dummy', 'LaunchPad', True, [])

### VESSEL ###

def get_new_vessels(past_vessels, current_vessels) -> list[Vessel]:
    new_vessels = list(set(current_vessels) - set(past_vessels))
    target_vessels = []
    for v in new_vessels:
        if v.type == UTIL_CONN.space_center.VesselType.debris:
            continue
        target_vessels.append(v)
    return target_vessels


def switch_to_vessel(v: Vessel | str):
    if isinstance(v, str):
        target = get_vessel_by_name(v)
    else:
        target = v
    if target:
        sc = UTIL_CONN.space_center
        sc.active_vessel = target
        # KSP有时切换后会莫名其妙多出无用分级
        # 如果当前分级没有激活部件:
        while not target.parts.in_stage(target.control.current_stage):
            target.control.activate_next_stage()
        return True
    return False


def get_vessel_by_name(name):
    vessels = UTIL_CONN.space_center.vessels
    for v in vessels:
        if v.name == name:
            return v
    raise ValueError(name)


def get_parts_in_stage_by_type(vessel: Vessel, target: str, stage: int) -> list[Part]:
    parts_in_stage = vessel.parts.in_stage(stage)
    part_list = []
    for i in parts_in_stage:
        if getattr(i, target):
            part_list.append(i)
    return part_list

### PART_ID ###

def temp_switch_to_vessel(func):
    def wrapper(*args, **kwargs):
        before = UTIL_CONN.space_center.active_vessel
        temp: Vessel = kwargs['vessel']
        if before.name != temp.name:
            switch_to_vessel(temp)
        res = func(*args, **kwargs)
        if before.name != temp.name:
            switch_to_vessel(before)
        return res
    return wrapper

def generate_pid():
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=12))

def get_part_id(part: Part):
    return _get_part_id(part, vessel=part.vessel)

@temp_switch_to_vessel
def _get_part_id(part: Part, vessel: Vessel):
    tags = part.tag.split()
    try:
        i = tags.index('--pid')
    except ValueError:
        return None
    return tags[i+1]

def assign_part_id(part: Part):
    if not part.tag:
        return
    return _assign_part_id(part, vessel=part.vessel)

@temp_switch_to_vessel
def _assign_part_id(part: Part, vessel: Vessel):
    _id = get_part_id(part)
    if _id is not None:
        return _id
    _id = generate_pid()
    part.tag = part.tag + ' --pid ' + _id
    return _id

def get_part_by_pid(v: Vessel, id: str):
    """按id寻找Part, 注意这将会搜寻整个vessel"""
#     return _get_part_by_pid(v, id, vessel=v)
# @temp_switch_to_vessel
# def _get_part_by_pid(v: Vessel, id: str, vessel: Vessel):
    cur = v.parts.root
    stack = [cur]
    while stack:
        cur = stack.pop()
        if get_part_id(cur) == id:
            return cur
        stack += cur.children
    return None

def get_root_part(spacecraft: Spacecraft) -> Part:
    return _get_root_part(spacecraft, vessel=spacecraft.vessel)

@temp_switch_to_vessel
def _get_root_part(spacecraft: Spacecraft, vessel: Vessel) -> Part:
    if spacecraft.docked_with is None:
        root = spacecraft.vessel.parts.root
    else:
        dp_ex = spacecraft.docking_port_ext_docked_at
        root = dp_ex.part.docking_port.docked_part
    return root
    
def get_tagged_children_parts(root: Part) -> list[Part]:
    """返回以root为根部件的所有有tag标记的part, 
    注意只会搜索spacecraft范围而非整个vessel"""
    return _get_tagged_children_parts(root, vessel=root.vessel)

@temp_switch_to_vessel
def _get_tagged_children_parts(root: Part, vessel: Vessel) -> list[Part]:
    from .part_extension import _part_tag_docopt
    res = []
    stack = [root]
    while stack:
        cur = stack.pop()
        if cur.tag:
            res.append(cur)
            if _part_tag_docopt(cur)['--docking_port']:
                # 不搜索与其对接的spacecraft
                for p in cur.children:
                    if p.docking_port:
                        continue
                    stack.append(p)
                continue
        stack += cur.children
    return res

def assign_part_to_exts(spacecraft: Spacecraft, ext_dict: dict[str, PartExt]):
    return _assign_part_to_exts(spacecraft, ext_dict, vessel=spacecraft.vessel)

@temp_switch_to_vessel
def _assign_part_to_exts(spacecraft: Spacecraft, ext_dict: dict[str, PartExt], vessel: Vessel):
    root = get_root_part(spacecraft)
    stack = [root]
    while stack:
        cur = stack.pop()
        part_id = get_part_id(cur)
        if part_id and part_id in ext_dict:
            ext_dict[part_id]._part = cur
        stack += cur.children

### ABORT ###

class AbortException(Exception):
    pass


def abort() -> None:
    try:
        UTIL_CONN.space_center.active_vessel.control.abort = True
        # TODO: quickload
    except krpc.error.RPCError:
        return


def abort_checker(func):
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if self.tasks.abort_flag:
            raise AbortException
        return result

    return wrapper


def add_abort_callback(task):
    def abort_callback(value):
        if value:
            task.tasks.abort_flag = True
            abort_stream.remove()
            task.tasks.task_queue.worker_lock.release()
            log = f'Task aborted:\n {task.short_description}'
            LOGGER.debug(log)

    if not task.conn:
        return
    abort_stream = task.conn.add_stream(getattr, task.vessel.control, 'abort')
    abort_stream.add_callback(abort_callback)
    abort_stream.start()
