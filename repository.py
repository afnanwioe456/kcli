import os
import json

from . import spacecrafts
from .spacecrafts import Spacecraft
from .task.tasks import TaskQueue, Tasks
from .utils import *

__all__ = [
    'save',
    'load',
]

CWD = os.getcwd()
REPO_DIR = CWD + "\\.live"

LOG_DIR = REPO_DIR + "\\log.txt"
HELPER_1_DIR = REPO_DIR + "\\helper.txt"
HELPER_2_DIR = REPO_DIR + "\\helper_2.txt"
COMMAND_HELP_DIR = REPO_DIR + "\\command_help.txt"
LAUNCH_HELP_DIR = REPO_DIR + "\\launch_help.txt"
QUEUE_DIR = REPO_DIR + "\\task_queue.txt"
SAVE_DIR = REPO_DIR + "\\save.json"


def _dump_all_spacecrafts():
    LOGGER.debug(f'Dumping all spacecrafts...')
    data = [s._to_dict() for s in Spacecraft._instances.values()]
    LOGGER.debug(f'Dumping complete.')
    return data
    

def _load_all_spacecrafts(data):
    LOGGER.debug(f'Loading all spacecrafts...')
    for d in data:
        cls = getattr(spacecrafts, d['_class_name'], None)
        if cls and issubclass(cls, Spacecraft):
            cls._from_dict(d)
        else:
            raise ValueError(f"Unknown or invalid class: {d['_class_name']}")
    LOGGER.debug(f'Assigning Part objects to each spacecraft...')
    for s in Spacecraft._instances.values():
        s: Spacecraft
        assign_part_to_exts(s, s.part_exts._all)
    LOGGER.debug(f'Loading complete.')


def _dump_all_tasks():
    LOGGER.debug(f'Dumping all tasks...')
    task_list = []
    with TaskQueue._queue_condition:
        while TaskQueue._size > 0:
            task_list.append(TaskQueue.get()._to_dict())
    LOGGER.debug(f'Dumping complete.')
    return task_list


def _load_all_tasks(data):
    LOGGER.debug(f'Loading all tasks...')
    if TaskQueue._size > 0:
        TaskQueue.clear()
    for t in data:
        TaskQueue.put(Tasks._from_dict(t))
    LOGGER.debug(f'Loading complete.')


def save():
    data =  {
        'spacecraft': _dump_all_spacecrafts(),
        'task_queue': _dump_all_tasks(),
    }
    with open(SAVE_DIR, 'w') as file:
        json.dump(data, file, indent=4)
    

def load():
    with open(SAVE_DIR, 'r') as file:
        data = json.load(file)
    _load_all_spacecrafts(data['spacecraft'])
    _load_all_tasks(data['task_queue'])


def write_helper(content: str):
    match content:
        case "command":
            with open(COMMAND_HELP_DIR, 'r', encoding='utf-8') as file:
                help_content = file.read()
        case "launch":
            with open(LAUNCH_HELP_DIR, 'r', encoding='utf-8') as file:
                help_content = file.read()
        case _:
            return


def write_task_queue(s: str):
    with open(QUEUE_DIR, 'w', encoding='utf-8') as file:
        file.write(s)

