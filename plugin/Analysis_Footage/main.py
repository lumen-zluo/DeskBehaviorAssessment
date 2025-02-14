import os
from datetime import datetime
from FuzzyDetection import variance_of_laplacian, gradient_based_blur_detection
import csv
import math
from ultralytics import YOLO
import numpy as np
from DrawTrajectory import DrawImage
import cv2
from scipy.ndimage import gaussian_filter
from DrawHeatMap import HeatMap
from tqdm import tqdm
from HomographyTransform import compute_homography_and_transform_gaze
import pandas as pd


def check_file(path):
    """
    检查里面三个文件是否存在
    """
    EyeVideoName = "1.mp4"
    GazeName = "1.txt"
    TimeStampName = "time.txt"

    files = os.listdir(path)
    for file in files:
        if file.endswith(".mp4_timestamp.txt"):
            TimeStampName = file
        if file.endswith(".mp4_gaze.txt"):
            GazeName = file
        if file.endswith(".mp4"):
            EyeVideoName = file

    Eye_path = os.path.join(path, EyeVideoName)
    Gaze_path = os.path.join(path, GazeName)
    TimeStamp_path = os.path.join(path, TimeStampName)
    file_list = [Eye_path, Gaze_path, TimeStamp_path]

    for i in file_list:

        if os.path.exists(i):

            continue

        else:

            return False

    return file_list

def convert_ndarray_to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return [convert_ndarray_to_list(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_ndarray_to_list(value) for key, value in data.items()}
    else:
        return data


def calculate_fixations(gaze_time, gaze_data, spatial_threshold=20, duration_threshold=100, start_index_offset=0):
        fixations = []
        current_fixation = []
        current_fixation_start = 0

        def euclidean_distance(point1, point2):
            return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

        for i, point in enumerate(gaze_data):
            if not current_fixation:
                # Start a new fixation
                current_fixation = [point]
                current_fixation_start = i
            else:
                # Check distance to the last point in the current fixation
                if euclidean_distance(current_fixation[-1], point) <= spatial_threshold:
                    current_fixation.append(point)
                else:
                    # Finalize the current fixation
                    duration = (i - current_fixation_start) * (1000 / 60.0)  # Duration in milliseconds
                    if duration >= duration_threshold:
                        fixation_x = sum(p[0] for p in current_fixation) / len(current_fixation)
                        fixation_y = sum(p[1] for p in current_fixation) / len(current_fixation)
                        if fixation_x > 0 and fixation_y > 0:
                            fixations.append({
                                'position': (int(fixation_x), int(fixation_y)),
                                'start_index': current_fixation_start + start_index_offset,
                                'started_at': gaze_time[current_fixation_start + start_index_offset],
                                'duration': duration
                            })
                    # Reset for new fixation
                    current_fixation = [point]
                    current_fixation_start = i
        # Final check for the last fixation
        if current_fixation:
            duration = (len(gaze_data) - current_fixation_start) * (1000 / 60.0)
            if duration >= duration_threshold:
                fixation_x = sum(p[0] for p in current_fixation) / len(current_fixation)
                fixation_y = sum(p[1] for p in current_fixation) / len(current_fixation)
                if fixation_x > 0 and fixation_y > 0:
                    fixations.append({
                        'position': (int(fixation_x), int(fixation_y)),
                        'start_index': current_fixation_start + start_index_offset,
                        'started_at': gaze_time[current_fixation_start + start_index_offset],
                        'duration': duration
                    })

        return fixations

def get_object_of_fixation(fixation):
    """
    获取注视点的物体
    """
    results = model.predict(fixation['img'])

    result = results[0]

    img = result.plot(boxes=False)

    boxes = result.boxes

    object_name = 'other'

    for i, box in enumerate(boxes):
        cords = box.xyxy[0].tolist()
        x1, y1, x2, y2 = cords
        class_id = int(box.cls[0].item())
        if x1 < fixation['position'][0] and y2 > fixation['position'][1]:
            # show object box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if class_id == index_to_pen:
                object_name = 'pen'
            elif class_id == index_to_ruler:
                object_name = 'ruler'
            elif class_id == index_to_worksheet:
                object_name = 'worksheet'
            elif class_id == index_to_eraser:
                object_name = 'eraser'

    return object_name
    

class FootageProcess:

    def __init__(self, rootPath, video_name, gaze_name, timestamp_name):

        self.save_root_path = None
        self.fixations = None
        self.end_index = None
        self.start_index = None
        self.whole_timestamp = None  # 储存整个timestamp列表
        self.root_path = rootPath
        self.video_path = os.path.join(self.root_path, video_name)
        self.gaze_path = os.path.join(self.root_path, gaze_name)
        self.time_path = os.path.join(self.root_path, timestamp_name)
        self.gaze_whole_list = []
        self.gaze_footage_list = []
        self.read_timestamp()
        self.read_gaze()

    def read_gaze(self):
        """
        读取眼睛数据文件
        """
        with open(self.gaze_path, "r") as file:
            gaze = file.readlines()
            self.gaze_whole_list = [[int(j) for j in i.strip().split(" ")] for i in gaze]

    def read_timestamp(self):
        """
        读取时间戳文件
        """
        time_format = "%Y-%m-%d %H:%M:%S.%f"

        with open(self.time_path, "r") as f:
            line = f.readlines()[1:]
            self.whole_timestamp = [datetime.strptime(i.strip(), time_format) for i in line]

    def find_best_timestamp(self, target_time):

        """
        找到目标时间与整个时间戳列表的最接近的时间戳
        """
        # 计算每个时间与目标时间的绝对差异（以秒为单位）
        differences = [abs((ts - target_time).total_seconds()) for ts in self.whole_timestamp]

        # 找出最小差异的值
        min_diff = min(differences)

        # 找到所有具有最小差异的索引（处理多个最接近的情况）
        min_diff_indices = [i for i, diff in enumerate(differences) if diff == min_diff]

        return min_diff_indices[0]

    def CutFootage(self, save_file_path, start_time=None, end_time=None, show=False):
        self.save_root_path = save_file_path
        time1 = None
        time2 = None
        if start_time is not None and end_time is not None:

            time_format = "%Y-%m-%d %H:%M:%S.%f"

            try:
                time1 = datetime.strptime(start_time, time_format)
                time2 = datetime.strptime(end_time, time_format)
            except ValueError as e:
                print(f"目标时间格式错误: {e}")
                exit()

            if time1 > time2:
                raise ValueError(f"start time为{start_time}, end time为{end_time}, 起始时间大于结束时间！,请检测")

        else:

            time1 = self.whole_timestamp[0]
            time2 = self.whole_timestamp[-1]

        self.start_index = self.find_best_timestamp(time1)
        self.end_index = self.find_best_timestamp(time2)
        print("start index:", self.start_index)
        print("end index:", self.end_index)
        self.gaze_footage_list = self.gaze_whole_list[self.start_index:self.end_index + 1]
        # print(self.gaze_footage_list)
        self.gaze_time_list = self.whole_timestamp[self.start_index:self.end_index + 1]
        self.fixations = calculate_fixations(self.gaze_time_list, self.gaze_footage_list, start_index_offset=self.start_index)
        # print(self.fixations)
        self.display_specific_frames()
        imgs = []
        current_img = []  # 当前场景的图像和注视点
        current_fixation = []  # 当前场景的图像和注视点
        duration_list = []
        for index, fixation in enumerate(self.fixations):
            if not current_img:
                current_img = [fixation['img']]
                current_fixation = [fixation['position']]
                duration_list = [fixation['duration']]
            else:
                img1 = current_img[0]  # 始终与第一张图像比较
                img2 = fixation['img']
                img2_gaze_position_x, img2_gaze_position_y = fixation['position']

                new_gaze_x, new_gaze_y, M = compute_homography_and_transform_gaze(img1, img2, (
                    img2_gaze_position_x, img2_gaze_position_y), show=False)

                if M is None or not (0 <= new_gaze_x < img1.shape[1] and 0 <= new_gaze_y < img1.shape[0]):
                    # 场景变化，保存当前结果
                    imgs.append({
                        'img': current_img[0],
                        'fixation_position': current_fixation,
                        'duration_list': duration_list
                    })
                    # 重置
                    current_img = [fixation['img']]
                    current_fixation = [fixation['position']]
                    duration_list = [fixation['duration']]
                else:
                    # 场景相同，累积 gaze 位置
                    current_img.append(img2)
                    current_fixation.append((new_gaze_x, new_gaze_y))
                    duration_list.append(fixation['duration'])

        # 保存最后结果
        if current_img and current_fixation:
            imgs.append({
                'img': current_img[0],
                'fixation_position': current_fixation,
                'duration_list': duration_list
            })

        self.save_ori_fixation_img(show=show, imgs=imgs)

        max_fixation_num = self.max_num_duration_whole(imgs)

        self.save_fixation_path(max_fixation_num, imgs)

        self.save_heatmap(imgs)

        cv2.destroyAllWindows()

        print("All done!!!!")

        return imgs

    def save_heatmap(self, imgs):

        heatmap_path = os.path.join(self.save_root_path, "heatmap")

        os.makedirs(heatmap_path, exist_ok=True)

        for index, img_dict in tqdm(enumerate(imgs), total=len(imgs), desc="Saving heatmap"):
            HeatMap(img_dict, heatmap_path, f"heatmap {index + 1}.jpg")

    def max_num_duration_whole(self, imgs):
        """
        这个是看看全部数据里面最大的注视点的数目有多少，方便后续计算注视点圆圈的大小
        """
        max_num = 0
        for i in imgs:
            if len(i['duration_list']) > max_num:
                max_num = len(i['duration_list'])
        return max_num

    def save_fixation_path(self, max_fixation_num, imgs):

        fixation_path = os.path.join(self.save_root_path, "fixation_path")

        os.makedirs(fixation_path, exist_ok=True)

        for index, img in tqdm(enumerate(imgs), total=len(imgs), desc="Saving fixation path"):
            sub_num_fixation = len(img['duration_list'])

            Range_list = [5, 5 + 10 * sub_num_fixation / max_fixation_num]

            DrawImage(Range_list, img, fixation_path, f"fixation {index + 1}.mp4")

    def save_ori_fixation_img(self, show, imgs):

        ori_fixation_img_path = os.path.join(self.save_root_path, "ori_fixation_img")

        os.makedirs(ori_fixation_img_path, exist_ok=True)

        # 显示结果
        num_pic = 1

        for img_dict in tqdm(imgs, total=len(imgs), desc="Saving original img"):
            img_save_path = os.path.join(ori_fixation_img_path, f"fixation {num_pic}.jpg")
            cv2.imwrite(img_save_path, img_dict['img'])
            num_pic += 1

            if show:
                cv2.imshow("img", img_dict['img'])
                cv2.waitKey(2000)

    def display_specific_frames(self):

        gaze_list = [i['position'] for i in self.fixations]

        frame_numbers = [i['start_index'] for i in self.fixations]
        # 打开视频文件
        cap = cv2.VideoCapture(self.video_path)

        # 检查视频是否成功打开
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        for index, frame_number in enumerate(frame_numbers):
            # 设置到特定帧
            index_right = False  # 这个是确认这一帧图片是否模糊
            gaze_position = gaze_list[index]
            frame = None
            sum_index = 1
            while not index_right and sum_index < 4:

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                # 读取该帧
                ret, frame = cap.read()
                VOL = variance_of_laplacian(frame)
                GBD = gradient_based_blur_detection(frame)

                if VOL or GBD:
                    index_right = True

                else:
                    sum_ = (-1) ** (sum_index) * math.ceil(sum_index / 2)
                    frame_number += sum_
                    sum_index += 1

            if frame is not None:
                # 显示该帧
                self.fixations[index]['img'] = frame
            else:
                raise ValueError(f"Error: Could not read frame {frame_number}.")

        # 释放视频捕获对象
        cap.release()
        cv2.destroyAllWindows()

    def save_object_detection(self, save_file_path, start_time=None, end_time=None, show=False):
        self.save_root_path = save_file_path
        time1 = None
        time2 = None
        if start_time is not None and end_time is not None:

            time_format = "%Y-%m-%d %H:%M:%S.%f"

            try:
                time1 = datetime.strptime(start_time, time_format)
                time2 = datetime.strptime(end_time, time_format)
            except ValueError as e:
                print(f"目标时间格式错误: {e}")
                exit()

            if time1 > time2:
                raise ValueError(f"start time为{start_time}, end time为{end_time}, 起始时间大于结束时间！,请检测")

        else:

            time1 = self.whole_timestamp[0]
            time2 = self.whole_timestamp[-1]

        self.start_index = self.find_best_timestamp(time1)
        self.end_index = self.find_best_timestamp(time2)
        print("start index:", self.start_index)
        print("end index:", self.end_index)
        self.gaze_footage_list = self.gaze_whole_list[self.start_index:self.end_index + 1]
        # print(self.gaze_footage_list)
        self.gaze_time_list = self.whole_timestamp[self.start_index:self.end_index + 1]
        self.fixations = calculate_fixations(self.gaze_time_list, self.gaze_footage_list, start_index_offset=self.start_index)
        # print(self.fixations)
        # self.display_specific_frames()

        # 打开视频文件
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')# 使用 mp4v 編碼
        os.makedirs(self.save_root_path, exist_ok=True)
        save_video_path = os.path.join(self.save_root_path, "fixations.mp4")
        wr = cv2.VideoWriter(save_video_path, fourcc, fps, (frame_width, frame_height))

        # 检查视频是否成功打开
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        frame_numbers = [i['start_index'] for i in self.fixations]
        max_frame = max(frame_numbers)
        flatten_fixations = []
        flag = 0

        for index, frame_number in enumerate(frame_numbers):
            next_start_index = frame_numbers[index + 1] if index + 1 < len(frame_numbers) else 0

            sub_items_count = next_start_index - frame_number

            for _ in range(sub_items_count):
                flatten_fixations.append(self.fixations[index])

        save_csv_path = os.path.join(self.save_root_path, "fixations.csv")
        csv_header = ['duration', 'fixation_x', 'fixation_y', 'object']
        with open(file=save_csv_path, mode="w", newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(csv_header)
            makeup_frame = total_frames - len(flatten_fixations)
            # print(len(flatten_fixations), total_frames)

            while True:
                ret, img = cap.read()

                results = model.predict(img)

                result = results[0]

                img = result.plot(boxes=False)

                boxes = result.boxes

                csv_data = [
                    flatten_fixations[flag]['duration'],
                    flatten_fixations[flag]['position'][0],
                    flatten_fixations[flag]['position'][1],
                    None
                ]

                for i, box in enumerate(boxes):
                    cords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = cords
                    class_id = int(box.cls[0].item())
                    if x1 < flatten_fixations[flag]['position'][0] and y2 > flatten_fixations[flag]['position'][1]:
                        # show object box
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        if class_id == index_to_pen:
                            csv_data[3] = 'pen'
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, 'pen', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        elif class_id == index_to_ruler:
                            csv_data[3] = 'ruler'
                            cv2.rectangle(img, (x1, y1), (x2, y2), (33, 37, 43), 2)
                            cv2.putText(img, 'ruler', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (33, 37, 43), 2)
                        elif class_id == index_to_worksheet:
                            csv_data[3] = 'worksheet'
                            cv2.rectangle(img, (x1, y1), (x2, y2), (233, 48, 95), 2)
                            cv2.putText(img, 'worksheet', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (233, 48, 95), 2)
                        elif class_id == index_to_eraser:
                            csv_data[3] = 'eraser'
                            cv2.rectangle(img, (x1, y1), (x2, y2), (95, 173, 101), 2)
                            cv2.putText(img, 'eraser', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (95, 173, 101), 2)

                cv2.circle(img, (int(flatten_fixations[flag]['position'][0]), int(flatten_fixations[flag]['position'][1])), 10, (95, 48, 233), 4)

                # cv2.imshow('img', img)
                wr.write(img)
                csv_writer.writerow(csv_data)

                if not flag == len(flatten_fixations) - 1:
                    flag += 1

                if not ret:
                    break
                # key = cv2.waitKey(0)
                # if key == ord('q'):
                #     break

        cap.release()
        wr.release()
        cv2.destroyAllWindows()
        print("All done!!!!")

        return []
    
    
    def CutFootageWithObject(self, save_file_path, start_time=None, end_time=None, show=False):
        self.save_root_path = save_file_path
        time1 = None
        time2 = None
        if start_time is not None and end_time is not None:

            time_format = "%Y-%m-%d %H:%M:%S.%f"

            try:
                time1 = datetime.strptime(start_time, time_format)
                time2 = datetime.strptime(end_time, time_format)
            except ValueError as e:
                print(f"目标时间格式错误: {e}")
                exit()

            if time1 > time2:
                raise ValueError(f"start time为{start_time}, end time为{end_time}, 起始时间大于结束时间！,请检测")

        else:

            time1 = self.whole_timestamp[0]
            time2 = self.whole_timestamp[-1]

        self.start_index = self.find_best_timestamp(time1)
        self.end_index = self.find_best_timestamp(time2)
        print("start index:", self.start_index)
        print("end index:", self.end_index)
        self.gaze_footage_list = self.gaze_whole_list[self.start_index:self.end_index + 1]
        # print(self.gaze_footage_list)
        self.gaze_time_list = self.whole_timestamp[self.start_index:self.end_index + 1]
        self.fixations = calculate_fixations(self.gaze_time_list, self.gaze_footage_list, start_index_offset=self.start_index)
        # print(self.fixations)
        self.display_specific_frames()
        imgs = []
        current_img = []  # 当前场景的图像和注视点
        current_fixation = []  # 当前场景的图像和注视点
        duration_list = []
        object_list = []
        started_list = []
        scene_id = 0
        for index, fixation in enumerate(self.fixations):
            fixation_object = get_object_of_fixation(fixation)
            if not current_img:
                current_img = [fixation['img']]
                current_fixation = [fixation['position']]
                duration_list = [fixation['duration']]
                object_list = [fixation_object]
                started_list = [fixation['started_at']]
            else:
                img1 = current_img[0]  # 始终与第一张图像比较
                img2 = fixation['img']
                img2_gaze_position_x, img2_gaze_position_y = fixation['position']

                new_gaze_x, new_gaze_y, M = compute_homography_and_transform_gaze(img1, img2, (
                    img2_gaze_position_x, img2_gaze_position_y), show=False)

                if M is None or not (0 <= new_gaze_x < img1.shape[1] and 0 <= new_gaze_y < img1.shape[0]):
                    # 场景变化，保存当前结果
                    scene_id += 1
                    imgs.append({
                        'img': current_img[0],
                        'fixation_position': current_fixation,
                        'duration_list': duration_list,
                        'scene_id': scene_id,
                        'object_list': object_list,
                        'started_list': started_list
                    })
                    # 重置
                    current_img = [fixation['img']]
                    current_fixation = [fixation['position']]
                    duration_list = [fixation['duration']]
                    object_list = [fixation_object]
                    started_list = [fixation['started_at']]
                else:
                    # 场景相同，累积 gaze 位置
                    current_img.append(img2)
                    current_fixation.append((new_gaze_x, new_gaze_y))
                    duration_list.append(fixation['duration'])
                    object_list.append(fixation_object)
                    started_list.append(fixation['started_at'])

        # 保存最后结果
        if current_img and current_fixation:
            scene_id += 1
            imgs.append({
                'img': current_img[0],
                'fixation_position': current_fixation,
                'duration_list': duration_list,
                'scene_id': scene_id,
                'object_list': object_list,
                'started_list': started_list
            })

        self.save_ori_fixation_img(show=show, imgs=imgs)

        max_fixation_num = self.max_num_duration_whole(imgs)

        self.save_fixation_path(max_fixation_num, imgs)

        self.save_heatmap(imgs)

        cv2.destroyAllWindows()

        print("All done!!!!")

        return imgs


if __name__ == '__main__':

    root_path = r"D:/DATA/DesktopBehaviorData"
    # root_path = r"D:/DesktopEyetrackerData"

    subjects = os.listdir(root_path)

    save_root_path = r"./output"

    for subject in subjects:
        subject_path = os.path.join(root_path, subject)

        file_list = check_file(subject_path)
        video_name = file_list[0]
        gaze_name = file_list[1]
        timestamp_name = file_list[2]

        if not file_list:
            raise ValueError("该文件夹没有所需要的文件，请检查！")

        project_name = f"{subject}"
        save_path = os.path.join(save_root_path, project_name)

        footage_process = FootageProcess(subject_path, video_name, gaze_name, timestamp_name)

        model_path = '../../model/testv3.pt'

        model = YOLO(model_path)
        index_to_ruler = 0
        index_to_worksheet = 1
        index_to_eraser = 2
        index_to_pen = 3

        # result = footage_process.save_object_detection(save_path)
        # result = footage_process.CutFootage(save_path)

        result = footage_process.CutFootageWithObject(save_path)
        json_file_name = 'fixations.csv'
        save_csv_path = os.path.join(save_path, json_file_name)

        csv_header = ['scene_id', 'started_list', 'duration_list', 'fixation_x', 'fixation_y', 'object_list']

        with open(file=save_csv_path, mode="w", newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(csv_header)
            for i in result:
                started_at = [dt.strftime('%Y-%m-%d %H:%M:%S.%f') for dt in i['started_list']]
                fixation_x = [float(x) for x, y in i['fixation_position']]
                fixation_y = [float(y) for x, y in i['fixation_position']]

                csv_data = [
                    i['scene_id'],
                    started_at,
                    i['duration_list'],
                    fixation_x,
                    fixation_y,
                    i['object_list'],
                ]
                csv_writer.writerow(csv_data)


