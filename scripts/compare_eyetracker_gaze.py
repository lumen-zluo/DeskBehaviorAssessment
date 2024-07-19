import os

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    root_path = r'E:\data'
    eyetracker_gaze_directories = os.listdir(root_path)
    # Total 10 non-adhd
    non_adhd_dic = ['subject26', 'subject27', 'subject29', 'subject31', 'subject38', 'subject40', 'subject45',
                   'subject46', 'subject48', 'subject58']

    adhd_dic = []
    for subject in eyetracker_gaze_directories:
        if subject not in non_adhd_dic:
            adhd_dic.append(subject)
    # randomly pick 10 adhd
    adhd_dic = random.sample(adhd_dic, 10)

    non_adhd_gaze = []
    adhd_gaze = []

    adhd_stability_index = []
    non_adhd_stability_index = []

    adhd_total_distance = []
    non_adhd_total_distance = []

    adhd_change_frequency = []
    non_adhd_change_frequency = []

    merge_subject = adhd_dic + non_adhd_dic
    for subject in merge_subject:
        directory_path = os.path.join(root_path, subject, 'pupil')
        for filename in os.listdir(directory_path):
            # if filename.endswith('.mp4_timestamp.txt') and not filename.startswith("."):
            #     time_file_path = os.path.join(directory_path, filename)
            #     # Read eyetracker time data
            #     if os.path.getsize(time_file_path) > 0:
            #         eyetracker_time = pd.read_csv(time_file_path, header=None, skiprows=1)
            #         eyetracker_timestamps = pd.to_datetime(eyetracker_time[0], format="%Y-%m-%d %H:%M:%S.%f")
            #         formatted_time = eyetracker_timestamps.dt.strftime("%H:%M:%S")
            if filename.endswith('.mp4_gaze.txt') and not filename.startswith("."):
                gaze_file_path = os.path.join(directory_path, filename)
                eyetracker_gaze = pd.read_csv(gaze_file_path, sep=' ', header=None, names=['x', 'y'])

        # 提取坐标点的 x 值和 y 值
        x_coordinates = eyetracker_gaze['x']
        y_coordinates = eyetracker_gaze['y']

        # 假设您有一个包含gaze数据的DataFrame，名为df，并且包含以下列：'x_coordinate'和'y_coordinate'
        final_combined_data = pd.DataFrame({'x_coordinate': x_coordinates, 'y_coordinate': y_coordinates})

        # 选择要归一化的列
        columns_to_normalize = ['x_coordinate', 'y_coordinate']

        # 对选定的列进行归一化
        final_combined_data[columns_to_normalize] = (final_combined_data[columns_to_normalize] - final_combined_data[columns_to_normalize].min()) / (
                    final_combined_data[columns_to_normalize].max() - final_combined_data[columns_to_normalize].min())

        # 计算眼动稳定性
        stability_x = final_combined_data['x_coordinate'].std()
        stability_y = final_combined_data['y_coordinate'].std()
        stability_index = (stability_x + stability_y) / 2 if not np.isnan(stability_x) and not np.isnan(
            stability_y) else 0

        # 计算眼动轨迹长度
        final_combined_data['dx'] = final_combined_data['x_coordinate'].diff()
        final_combined_data['dy'] = final_combined_data['y_coordinate'].diff()
        final_combined_data['distance'] = np.sqrt(final_combined_data['dx'] ** 2 + final_combined_data['dy'] ** 2)
        total_distance = final_combined_data['distance'].sum()

        # 计算凝视点变化频率
        changes = (final_combined_data['dx'] != 0) | (final_combined_data['dy'] != 0)
        change_frequency = changes.sum() / len(final_combined_data)

        if subject in non_adhd_dic:
            non_adhd_stability_index.append(stability_index)
            non_adhd_total_distance.append(total_distance)
            non_adhd_change_frequency.append(change_frequency)
        else:
            adhd_stability_index.append(stability_index)
            adhd_total_distance.append(total_distance)
            adhd_change_frequency.append(change_frequency)



        # 绘制分布图
        # if subject in non_adhd_dic:
        #     plt.title('Non-adhd Gaze')
        #     plt.scatter(final_combined_data['x_coordinate'], final_combined_data['y_coordinate'], label=subject, c='blue')
        # else:
        #     plt.title('Adhd Gaze')
        #     plt.scatter(final_combined_data['x_coordinate'], final_combined_data['y_coordinate'], label=subject, c='orange')
        #
        # # 添加图例和标签
        # plt.legend()
        # plt.xlabel("X Coordinate")
        # plt.ylabel("Y Coordinate")
        #
        # file_name = fr"D:\Analyse\Desktop Behavior\{subject}_distribution.png"
        # plt.savefig(file_name)
        # plt.clf() # Clean all result

    # 创建第一个子图，并绘制眼动稳定性数据
    plt.subplot(3, 2, 1)  # 创建第一个子图
    plt.plot(range(0, len(adhd_stability_index)), adhd_stability_index, 'bo')  # 绘制眼动稳定性数据
    plt.xlabel('Subject')
    plt.ylim(-0.5, 0.5)
    # plt.ylabel('ADHD Eye Stability')
    plt.title('Eye Stability of ADHD')

    # 创建第二个子图，并绘制注视点移动长度数据
    plt.subplot(3, 2, 2)  # 创建第二个子图
    plt.plot(range(0, len(non_adhd_stability_index)), non_adhd_stability_index, 'ro')  # 绘制注视点移动长度数据
    plt.xlabel('Subject')
    plt.ylim(-0.5, 0.5)
#     plt.ylabel('Non-Adhd Eye Stability')
    plt.title('Eye Stability of Non-ADHD')

    # 调整子图之间的间距
    plt.tight_layout()

    # 创建第一个子图，并绘制眼动稳定性数据
    plt.subplot(3, 2, 3)  # 创建第一个子图
    plt.plot(range(0, len(adhd_change_frequency)), adhd_change_frequency, 'bo')  # 绘制眼动稳定性数据
    plt.xlabel('Subject')
    plt.ylim(0, 1)
#     plt.ylabel('ADHD Change Frequency')
    plt.title('Change Frequency of ADHD')

    # 创建第二个子图，并绘制注视点移动长度数据
    plt.subplot(3, 2, 4)  # 创建第二个子图
    plt.plot(range(0, len(non_adhd_change_frequency)), non_adhd_change_frequency, 'ro')  # 绘制注视点移动长度数据
    plt.xlabel('Subject')
    plt.ylim(0, 1)
#     plt.ylabel('Non-Adhd Change Frequency')
    plt.title('Change Frequency of Non-ADHD')

    # 调整子图之间的间距
    plt.tight_layout()

    # 创建第一个子图，并绘制眼动稳定性数据
    plt.subplot(3, 2, 5)  # 创建第一个子图
    plt.plot(range(0, len(adhd_total_distance)), adhd_total_distance, 'bo')  # 绘制眼动稳定性数据
    plt.xlabel('Subject')
    plt.ylim(-1000, 3500)
#     plt.ylabel('ADHD Total Distance')
    plt.title('Total Distance of ADHD')

    # 创建第二个子图，并绘制注视点移动长度数据
    plt.subplot(3, 2, 6)  # 创建第二个子图
    plt.plot(range(0, len(non_adhd_total_distance)), non_adhd_total_distance, 'ro')  # 绘制注视点移动长度数据
    plt.xlabel('Subject')
    plt.ylim(-1000, 3500)
#     plt.ylabel('Non-Adhd Total Distance')
    plt.title('Total Distance of Non-ADHD')

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()




    # take as a whole group
    for subject in eyetracker_gaze_directories:
        directory_path = os.path.join(root_path, subject, 'pupil')
        for filename in os.listdir(directory_path):
            if filename.endswith('.mp4_gaze.txt') and not filename.startswith("."):
                gaze_file_path = os.path.join(directory_path, filename)
                eyetracker_gaze = pd.read_csv(gaze_file_path, sep=' ', header=None, names=['x', 'y'])
                if subject in non_adhd_dic:
                    non_adhd_gaze.append(eyetracker_gaze)
                else:
                    adhd_gaze.append(eyetracker_gaze)

                # 提取坐标点的 x 值和 y 值
                x_coordinates = eyetracker_gaze['x']
                y_coordinates = eyetracker_gaze['y']

                # 假设您有一个包含gaze数据的DataFrame，名为df，并且包含以下列：'x_coordinate'和'y_coordinate'
                final_combined_data = pd.DataFrame({'x_coordinate': x_coordinates, 'y_coordinate': y_coordinates})

                # 选择要归一化的列
                columns_to_normalize = ['x_coordinate', 'y_coordinate']

                # 对选定的列进行归一化
                final_combined_data[columns_to_normalize] = (final_combined_data[columns_to_normalize] -
                                                             final_combined_data[columns_to_normalize].min()) / (
                                                                    final_combined_data[columns_to_normalize].max() -
                                                                    final_combined_data[columns_to_normalize].min())

                # 计算眼动稳定性
                stability_x = final_combined_data['x_coordinate'].std()
                stability_y = final_combined_data['y_coordinate'].std()
                stability_index = (stability_x + stability_y) / 2 if not np.isnan(stability_x) and not np.isnan(
                    stability_y) else 0

                # 计算眼动轨迹长度
                final_combined_data['dx'] = final_combined_data['x_coordinate'].diff()
                final_combined_data['dy'] = final_combined_data['y_coordinate'].diff()
                final_combined_data['distance'] = np.sqrt(
                    final_combined_data['dx'] ** 2 + final_combined_data['dy'] ** 2)
                total_distance = final_combined_data['distance'].sum()

                # 计算凝视点变化频率
                changes = (final_combined_data['dx'] != 0) | (final_combined_data['dy'] != 0)
                change_frequency = changes.sum() / len(final_combined_data)

                if subject in non_adhd_dic:
                    non_adhd_stability_index.append(stability_index)
                    non_adhd_total_distance.append(total_distance)
                    non_adhd_change_frequency.append(change_frequency)
                else:
                    adhd_stability_index.append(stability_index)
                    adhd_total_distance.append(total_distance)
                    adhd_change_frequency.append(change_frequency)

    # when take data as a whole group
    # 计算每组数据的平均值
    adhd_stability_index_avg = np.mean(adhd_stability_index)
    non_adhd_stability_index_avg = np.mean(non_adhd_stability_index)

    adhd_total_distance_avg = np.mean(adhd_total_distance)
    non_adhd_total_distance_avg = np.mean(non_adhd_total_distance)

    adhd_change_frequency_avg = np.mean(adhd_change_frequency)
    non_adhd_change_frequency_avg = np.mean(non_adhd_change_frequency)

    # 设置柱状图的位置和宽度
    x = np.arange(2)  # 有两组数据类型
    bar_width = 0.3  # 柱状图的宽度

    # 绘制 Stability Index 图表
    plt.subplot(1, 3, 1)
    plt.bar(x, [adhd_stability_index_avg, non_adhd_stability_index_avg], bar_width)
    plt.xticks(x, ['ADHD', 'Non-ADHD'])
    plt.title('Stability Index')

    # 绘制 Total Distance 图表
    plt.subplot(1, 3, 2)
    plt.bar(x, [adhd_total_distance_avg, non_adhd_total_distance_avg], bar_width, color='orange')
    plt.xticks(x, ['ADHD', 'Non-ADHD'])
    plt.title('Total Distance')

    # 绘制 Change Frequency 图表
    plt.subplot(1, 3, 3)
    plt.bar(x, [adhd_change_frequency_avg, non_adhd_change_frequency_avg], bar_width, color='red')
    plt.xticks(x, ['ADHD', 'Non-ADHD'])
    plt.title('Change Frequency')

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图表
    plt.show()

    # 最大注視區域
    non_adhd_visible_area = []
    adhd_visible_area = []
    for gaze in non_adhd_gaze:
        min_x = min(gaze['x'])
        max_x = max(gaze['x'])
        min_y = min(gaze['y'])
        max_y = max(gaze['y'])
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        non_adhd_visible_area.append(area)

    # 计算Non-ADHD组数据的范围
    for gaze in adhd_gaze:
        min_x = min(gaze['x'])
        max_x = max(gaze['x'])
        min_y = min(gaze['y'])
        max_y = max(gaze['y'])
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        adhd_visible_area.append(area)

    # 创建数据列表
    data = [adhd_visible_area, non_adhd_visible_area]

    # 创建标签
    labels = ['ADHD', 'Non-ADHD']

    # 绘制箱线图
    plt.boxplot(data, labels=labels)

    # 设置图表标题和轴标签
    plt.title('Comparison of Visual Area between ADHD and Non-ADHD Children')
    plt.xlabel('Group')
    plt.ylabel('Visual Area')

    # 显示图表
    plt.show()

