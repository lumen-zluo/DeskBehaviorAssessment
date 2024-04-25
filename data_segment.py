import os
from util.common import create_file_if_not_exists, convert_to_datetime
import csv
import pandas as pd
import statistics
import math
from datetime import datetime


class DataSegment:

    def __init__(self, filepath):

        self.synchronize_filepath = os.path.join(filepath, 'synchronize')
        self.segment_filepath = os.path.join(filepath, 'segment')
        self.filepath = filepath

        # raw data set path
        self.realsense_time_path = None
        self.realsense_joint_path = None

        self.eyetracker_time_path = None
        self.eyetracker_video_path = None
        self.eyetracker_gaze_path = None
        self.eyetracker_seg_index = None

        self.handwriting_time_path = None

        # data set path after process
        self.video_seg_root_path = None

        self.eyetracker_txt_time_list = None
        self.realsense_txt_time_list = None
        self.handwriting_txt_time_list = None
        self.keyframe_list = None

    def set_file_path(self):
        # csv_file_path = self.find_csv_file(realsense_time_path)
        self.realsense_time_path = os.path.join(self.filepath, "realsense", "realsense_time.txt")
        self.realsense_joint_path = os.path.join(self.filepath, "realsense", "joint.csv")

        self.eyetracker_time_path = os.path.join(self.filepath, "eyetracker", "time.txt")
        self.eyetracker_video_path = os.path.join(self.filepath, "eyetracker", "video.mp4")
        self.eyetracker_gaze_path = os.path.join(self.filepath, "eyetracker", "gaze.txt")

        self.handwriting_time_path = os.path.join(self.filepath, "handwriting", "handwriting_time.txt")

    def create_seg_file(self):
        create_file_if_not_exists(self.synchronize_filepath)
        create_file_if_not_exists(self.segment_filepath)

    def run_seg(self):
        print('Start data processing: {}'.format(self.filepath))
        self.set_file_path()
        self.create_seg_file()
        # df = pd.read_csv(self.realsense_time)
        realsense_time = open(self.realsense_time_path, 'r', encoding='utf-8')
        eyetracker_time = open(self.eyetracker_time_path, 'r', encoding="utf-8")
        handwriting_time = open(self.handwriting_time_path, 'r', encoding="utf-8")

        self.eyetracker_txt_time_list = [convert_to_datetime(line[-13:-1]) for line in eyetracker_time]
        self.realsense_txt_time_list = [convert_to_datetime(line[-13:-1]) for line in realsense_time]
        self.handwriting_txt_time_list = [convert_to_datetime(line[11:25]) for line in handwriting_time]
        realsense_time.close()
        eyetracker_time.close()
        handwriting_time.close()

        realsense_keys = []
        for timestamp in self.realsense_txt_time_list:
            # min_diff return 0 -> microseconds, 1 -> index
            min_diff = min((abs(time - timestamp), index) for index, time in enumerate(self.eyetracker_txt_time_list))
            realsense_keys.append(min_diff[1])

        handwriting_keys = []
        for timestamp in self.handwriting_txt_time_list:
            # min_diff return 0 -> microseconds, 1 -> index
            min_diff = min((abs(time - timestamp), index) for index, time in enumerate(self.eyetracker_txt_time_list))
            handwriting_keys.append(min_diff[1])

        # Read eyetracker gaze data
        eyetracker_gaze = []
        with open(self.eyetracker_gaze_path, 'r') as file:
            for line in file:
                line = line.strip()
                items = line.split()
                eyetracker_gaze.append(items)

        # Read handwriting data
        handwriting_data = []
        strokes_data = []
        temp = []
        idx = 0
        with open(self.handwriting_time_path, 'r') as file:
            for line in file:
                line = line.strip()
                row = [item.strip() for item in line.split(',')]
                pressure = row[3]
                handwriting_data.append(row)

                if pressure == '0':
                    if temp:
                        strokes_data.append(temp)
                        temp = []
                else:
                    temp.append([idx] + row)
                    idx += 1
            if temp:
                strokes_data.append(temp)

        # Read eyetracker time data
        eyetracker_time = []
        with open(self.eyetracker_time_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                eyetracker_time.append(line)

        # Read realsense skeleton data
        with open(self.realsense_joint_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            joint_data = list(reader)

        # Calculate Eyetracker Data
        # length
        eyetracker_total_length = 0
        distances = []
        for i in range(1, len(eyetracker_gaze)):
            x1, y1 = eyetracker_gaze[i - 1]
            x2, y2 = eyetracker_gaze[i]
            distance = math.sqrt((int(x2) - int(x1)) ** 2 + (int(y2) - int(y1)) ** 2)
            distances.append(distance)
            eyetracker_total_length += distance

        # Avg speed
        time_format = "%Y-%m-%d %H:%M:%S.%f"
        start_time = datetime.strptime(eyetracker_time[0], time_format)
        end_time = datetime.strptime(eyetracker_time[-1], time_format)
        eyetracker_total_duration = end_time - start_time
        eyetracker_ms = eyetracker_total_duration.total_seconds() * 1000
        eyetracker_avg_speed = eyetracker_total_length / eyetracker_ms

        # Std speed
        time_objects = [datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f") for time in eyetracker_time]
        time_diffs = [time_objects[i] - time_objects[i - 1] for i in range(1, len(time_objects))]
        time_diffs_ms_seconds = [time.total_seconds()*1000 for time in time_diffs]
        speeds = [distance / time_diff for distance, time_diff in zip(distances, time_diffs_ms_seconds)]
        std_deviation = statistics.stdev(speeds)

        # Calculate Handwriting Data
        # 壓力指標
        pressure_changes = []
        for i in range(1, len(handwriting_data)):
            current_pressure = handwriting_data[i][3]
            previous_pressure = handwriting_data[i - 1][3]
            pressure_change = abs(int(current_pressure) - int(previous_pressure))
            pressure_changes.append(pressure_change)
        # 计算平均速度和标准差
        pressure_average_speed = sum(pressure_changes) / len(pressure_changes)
        pressure_std_deviation = math.sqrt(
            sum((x - pressure_average_speed) ** 2 for x in pressure_changes) / len(pressure_changes)
        )

        # 運動指標
        handwriting_total_length = 0
        distances = []
        for i in range(1, len(handwriting_data)):
            x1 = handwriting_data[i - 1][1]
            y1 = handwriting_data[i - 1][2]
            x2 = handwriting_data[i][1]
            y2 = handwriting_data[i][2]
            distance = math.sqrt((int(x2) - int(x1)) ** 2 + (int(y2) - int(y1)) ** 2)
            distances.append(distance)
            handwriting_total_length += distance
        # Avg speed
        start_time = datetime.strptime(handwriting_data[0][0], time_format)
        end_time = datetime.strptime(handwriting_data[-1][0], time_format)
        handwriting_total_duration = end_time - start_time
        handwriting_ms = handwriting_total_duration.total_seconds() * 1000
        handwriting_avg_speed = handwriting_total_length / handwriting_ms

        # Std speed
        extracted_handwriting_time = [row[0] for row in handwriting_data]
        time_objects = [datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f") for time in extracted_handwriting_time]
        time_diffs = [time_objects[i] - time_objects[i - 1] for i in range(1, len(time_objects))]
        time_diffs_ms_seconds = [time.total_seconds() * 1000 for time in time_diffs]
        speeds = [distance / time_diff if time_diff != 0 else 0 for distance, time_diff in
                  zip(distances, time_diffs_ms_seconds)]
        handwriting_std_deviation = statistics.stdev(speeds)

        # Start Synchronizing
        output_path = os.path.join(self.synchronize_filepath, 'data.csv')
        with open(file=output_path, mode="w", newline='') as f:
            # Construct csv
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Timestamp', 'gaze_x', 'gaze_y', 'handwriting_x', 'handwriting_y',
                                 'pressure', 'left_elbow_angle', 'right_elbow_angle', 'left_neck_angle',
                                 'right_neck_angle', 'keypoint'])

            for index, timestamp in enumerate(eyetracker_time):
                gaze1 = eyetracker_gaze[index][0]
                gaze2 = eyetracker_gaze[index][1]

                if index in handwriting_keys:
                    key = handwriting_keys.index(index)
                    x = handwriting_data[key][1]
                    y = handwriting_data[key][2]
                    pressure = handwriting_data[key][3]
                else:
                    x = None
                    y = None
                    pressure = None

                if index in realsense_keys:
                    key = realsense_keys.index(index)
                    left_elbow_angle = joint_data[key][1]
                    right_elbow_angle = joint_data[key][2]
                    left_neck_angle = joint_data[key][3]
                    right_neck_angle = joint_data[key][4]
                    position = joint_data[key][5]
                else:
                    left_elbow_angle = None
                    right_elbow_angle = None
                    left_neck_angle = None
                    right_neck_angle = None
                    position = None

                csv_writer.writerow([timestamp, gaze1, gaze2, x, y, pressure, left_elbow_angle, right_elbow_angle,
                                     left_neck_angle, right_neck_angle, position])

        # Start Segment
        eyetracker_keys = []
        realsense_keys = []
        for timestamp in self.handwriting_txt_time_list:
            # min_diff return 0 -> microseconds, 1 -> index
            min_diff1 = min((abs(time - timestamp), index) for index, time in enumerate(self.eyetracker_txt_time_list))
            min_diff2 = min((abs(time - timestamp), index) for index, time in enumerate(self.realsense_txt_time_list))
            eyetracker_keys.append(min_diff1[1])
            realsense_keys.append(min_diff2[1])

        stroke_count = len(strokes_data)
        for index, stroke in enumerate(strokes_data):
            filename = f"stroke{index + 1}.csv"
            output_path = os.path.join(self.segment_filepath, filename)
            with open(file=output_path, mode="w", newline='') as f:
                # Construct csv
                csv_writer = csv.writer(f)
                csv_writer.writerow(['Timestamp', 'gaze_x', 'gaze_y', 'handwriting_x', 'handwriting_y',
                                     'pressure', 'left_elbow_angle', 'right_elbow_angle', 'left_neck_angle',
                                     'right_neck_angle', 'keypoint'])
                for point in stroke:
                    key = point[0]
                    eyetracker_key = eyetracker_keys[key]
                    realsense_key = realsense_keys[key]

                    gaze1 = eyetracker_gaze[eyetracker_key][0]
                    gaze2 = eyetracker_gaze[eyetracker_key][1]

                    timestamp = point[1]
                    x = point[2]
                    y = point[3]
                    pressure = point[4]

                    left_elbow_angle = joint_data[realsense_key][1]
                    right_elbow_angle = joint_data[realsense_key][2]
                    left_neck_angle = joint_data[realsense_key][3]
                    right_neck_angle = joint_data[realsense_key][4]
                    position = joint_data[realsense_key][5]

                    csv_writer.writerow([timestamp, gaze1, gaze2, x, y, pressure, left_elbow_angle, right_elbow_angle,
                                         left_neck_angle, right_neck_angle, position])

        print('Data processed successfully!')
