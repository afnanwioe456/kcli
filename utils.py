from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Type
from dateutil import parser, tz
from datetime import datetime, timedelta
import logging

import krpc
import krpc.services

from .repository import *

if TYPE_CHECKING:
    from krpc.services.spacecenter import Vessel, Part

KSP_EPOCH_TIME = -599616000
LAUNCH_SITES_COORDS = {
    'wenchang': (19.613726150307052, 110.9553275138089)
}

### LOGGER ###

def setup_logger():
    logger = logging.getLogger("krpclive")
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("krpc_live/.live/log.txt", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    debug_file_handler = logging.FileHandler("krpc_live/krpclive.log", encoding="utf-8")
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

UTIL_CONN = None
UTIL_CONN = krpc.connect('krpclive', address='127.0.0.1', rpc_port=65534, stream_port=65535)

### IN-GAME TIME CONTROL ###

def get_ut():
    return UTIL_CONN.space_center.ut


def time_wrap(start_time):
    ut = UTIL_CONN.space_center.ut
    if ut < start_time:
        LOGGER.debug(f'time wrapping to {sec_to_date(start_time)} ...')
        UTIL_CONN.space_center.warp_to(start_time)


def date_to_sec(input_date: str) -> int | None:
    """
    日期转换为自1951-01-01 00:00:00的秒数, 无法解析时返回None
    """
    try:
        # 尝试使用dateutil.parser解析输入
        parsed_date = parser.parse(input_date).replace(tzinfo=tz.tzutc())
        # 如果没有输入日期
        if parsed_date.date() == parser.parse('00:00:00').date():
            default_date = str(sec_to_date(int(get_ut())).date())
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


def sec_to_date(seconds) -> datetime:
    """
    自1951-01-01 00:00:00的秒数转换为datetime对象
    """
    epoch = datetime(1951, 1, 1, 0, 0, 0)
    time_difference = timedelta(seconds=int(seconds))
    result_datetime = epoch + time_difference
    return result_datetime

### LAUNCH_SITE ###

def get_launch_site_position(site: str = 'wenchang'):
    la, lo = LAUNCH_SITES_COORDS[site]
    body = UTIL_CONN.space_center.bodies['Earth']
    site_p = body.surface_position(la, lo, body.non_rotating_reference_frame)
    site_p = (site_p[0], site_p[2], site_p[1])
    return site_p

### VESSEL.NAME ###

def vessel_namer(name: str,
                 neg: list[str] = None) -> str:
    if neg is None:
        neg = []
    sc = UTIL_CONN.space_center
    vessels = sc.vessels
    max_count = 0  # 最高#尾编号
    for v in vessels:
        if v.name in neg:
            continue
        if v.type == sc.VesselType.debris:
            continue
        name_words = v.name.split('#')
        if name == ''.join(name_words[:-1]) or name == v.name:  # 是否重名
            try:  # 更新最大尾编号
                num = int(name_words[-1])
            except ValueError:
                num = 0
            if num > max_count:
                max_count = num + 1
    if not max_count:
        return name
    return f'{name}#{max_count}'


def get_original_name(name) -> str:
    words = name.split('#')
    try:
        int(words[-1])
        return '#'.join(words[:-1])
    except ValueError:
        return name

### VESSEL ###

def get_new_vessels(past_vessels, current_vessels) -> list[Vessel]:
    new_vessels = list(set(current_vessels) - set(past_vessels))
    target_vessels = []
    for v in new_vessels:
        if v.type == UTIL_CONN.space_center.VesselType.debris:
            continue
        target_vessels.append(v)
    return target_vessels


def switch_to_vessel(name):
    sc = UTIL_CONN.space_center
    target = get_vessel_by_name(name)
    if target:
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
    LOGGER.debug(f'{name}: 载具不存在!')


def get_parts_in_stage_by_type(vessel: Vessel, target: str, stage: int) -> list[Part]:
    parts_in_stage = vessel.parts.in_stage(stage)
    part_list = []
    for i in parts_in_stage:
        if getattr(i, target):
            part_list.append(i)
    return part_list

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
