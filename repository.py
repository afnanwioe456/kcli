import os
import json

from .spacecrafts import SpacecraftBase
from .task.tasks import TaskQueue

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


def save(task_queue: TaskQueue):
    data =  {
        'spacecraft': SpacecraftBase.dump_all(),
        'task_queue': task_queue.dump_all(),
    }
    with open(SAVE_DIR, 'w') as file:
        json.dump(data, file, indent=4)
    

def load():
    with open(SAVE_DIR, 'r') as file:
        data = json.load(file)
    SpacecraftBase.load_all(data['spacecraft'])
    return TaskQueue.load_all(data['task_queue'])


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

