import os
import time
from datetime import datetime


def get_time():
    now = time.localtime()
    now_time = time.strftime("%Y-%m-%d %H:%M:%S", now)
    return now_time


def create_file_if_not_exists(file_path):
    if not os.path.exists(file_path):
        print(file_path, " is not exists")
        os.makedirs(file_path)
    else:
        print(file_path, " is exists")
        pass


def find_csv_file(root_path):
    for file in os.listdir(root_path):
        if file.endswith('.csv'):
            return os.path.join(root_path, file)
    return None


def convert_to_datetime(time_str):
    return datetime.strptime(time_str, "%H:%M:%S.%f")


def check_file_exists(file_path):
    if os.path.exists(file_path):
        # print(f"{file_path} exist")
        return True
    else:
        # print(f"{file_path} not exist")
        return False
