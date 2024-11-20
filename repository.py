import os
from datetime import datetime

CWD = os.getcwd()
REPO_DIR = CWD + "\\.live"

LOG_DIR = REPO_DIR + "\\log.txt"
HELPER_1_DIR = REPO_DIR + "\\helper.txt"
HELPER_2_DIR = REPO_DIR + "\\helper_2.txt"
COMMAND_HELP_DIR = REPO_DIR + "\\command_help.txt"
LAUNCH_HELP_DIR = REPO_DIR + "\\launch_help.txt"
QUEUE_DIR = REPO_DIR + "\\task_queue.txt"


def write_log(log_message):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"###{current_time}:\n{log_message}\n"
    with open(LOG_DIR, 'a', encoding='utf-8') as log_file:
        log_file.write(log_entry)
    print(log_message)


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


if __name__ == '__main__':
    write_helper("command")
