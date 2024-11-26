import os
from datetime import datetime
from FuzzyDetection import variance_of_laplacian, gradient_based_blur_detection
import cv2
import math
import numpy as np
from DrawTrajectory import DrawImage
import cv2
from scipy.ndimage import gaussian_filter
from DrawHeatMap import HeatMap
from tqdm import tqdm
from HomographyTransform import compute_homography_and_transform_gaze


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

    return True


def calculate_fixations(gaze_data, spatial_threshold=20, duration_threshold=100, start_index_offset=0):

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
                    'duration': duration
                })

    return fixations


class FootageProcess:

    def __init__(self, rootPath):

        self.save_root_path = None
        self.fixations = None
        self.end_index = None
        self.start_index = None
        self.whole_timestamp = None  # 储存整个timestamp列表
        self.root_path = rootPath
        self.video_path = os.path.join(self.root_path, "1.mp4")
        self.gaze_path = os.path.join(self.root_path, "1.txt")
        self.time_path = os.path.join(self.root_path, "time.txt")
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

    def CutFootage(self, save_file_path, start_time = None, end_time = None, show = False):
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
        self.fixations = calculate_fixations(self.gaze_footage_list, start_index_offset=self.start_index)
        # print(self.fixations)
        self.display_specific_frames()
        imgs = []
        current_img = []  # 当前场景的图像和注视点
        current_fixation = []  # 当前场景的图像和注视点
        duration_list =[]
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
                img2_gaze_position_x, img2_gaze_position_y),show=False)

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

        self.save_fixation_path(max_fixation_num,imgs)

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

        for index, img in tqdm(enumerate(imgs),total=len(imgs),desc="Saving fixation path"):

            sub_num_fixation = len(img['duration_list'])

            Range_list = [5, 5 + 10 * sub_num_fixation / max_fixation_num]

            DrawImage(Range_list, img, fixation_path, f"fixation {index+1}.mp4")

    def save_ori_fixation_img(self, show, imgs):

        ori_fixation_img_path = os.path.join(self.save_root_path, "ori_fixation_img")

        os.makedirs(ori_fixation_img_path, exist_ok=True)

        # 显示结果
        num_pic = 1

        for img_dict in tqdm(imgs,total=len(imgs),desc="Saving original img"):
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


if __name__ == '__main__':

    root_path = r"D:DesktopEyetrackerVideo"

    subjects = os.listdir(root_path)

    save_root_path = r"./output"

    for subject in subjects:
        subject_path = os.path.join(root_path, subject)

        if not check_file(subject_path):
            raise ValueError("该文件夹没有所需要的文件，请检查！")

        project_name = f"{subject}"
        save_path = os.path.join(save_root_path, project_name)

        footage_process = FootageProcess(root_path)
        result = footage_process.CutFootage(save_path)

