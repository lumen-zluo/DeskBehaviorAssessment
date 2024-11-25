import os.path

import cv2
import numpy as np
import math


class DrawImage:

    def __init__(self, range_list, img, save_root, video_name):
        """
        {
        'img': img
        'fixation_position': fixation_position  is a list like [(x,y), (x,y), (x,y), (x,y)]
        'duration_list': duration_list is a list like [1, 2, 3, 4]
        }
        """
        self.range_list = range_list
        self.orig_img = img['img']
        self.fixation_position = img['fixation_position']
        self.duration_list = img['duration_list']
        self.frame_width = self.orig_img.shape[1]
        self.frame_height = self.orig_img.shape[0]
        self.fps = 1
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 'mp4v' 编码
        self.save_path = os.path.join(save_root, f'{video_name}.mp4')
        self.output = cv2.VideoWriter(self.save_path , fourcc, self.fps, (self.frame_width, self.frame_height))
        self.draw_trajectory()

    def add_text(self, img, text, position):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # 获取文本尺寸和基线
        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size

        # 确保文本在图像内
        text_x, text_y = position

        # 计算边框与文本之间的间距
        padding_x = 10  # 水平方向的填充
        padding_y = 10  # 垂直方向的填充

        # 计算矩形框的顶点坐标，确保文本在矩形框内部居中
        box_top_left = (text_x, text_y - text_height - padding_y)
        box_bottom_right = (text_x + text_width + 2 * padding_x, text_y + baseline + padding_y)

        # 画黑色边框的矩形
        cv2.rectangle(img,
                      box_top_left,
                      box_bottom_right,
                      (0, 0, 0),  # 黑色边框
                      thickness=2)  # 边框线条宽度

        # 在矩形内部填充白色
        cv2.rectangle(img,
                      (text_x + 2, text_y - text_height - padding_y + 2),  # 内缩2个像素以避免覆盖边框
                      (text_x + text_width + padding_x - 2, text_y + baseline + padding_y - 2),
                      (255, 255, 255),  # 白色填充
                      -1)

        # 计算文本的左上角坐标，使其在矩形内部居中
        text_start_x = text_x + padding_x
        text_start_y = text_y

        # 添加黑色文本
        cv2.putText(img,
                    text,
                    (text_start_x, text_start_y),
                    font,
                    font_scale,
                    (0, 0, 0),  # 黑色文本
                    thickness,
                    cv2.LINE_AA)

        return img

    def find_position_away_from_lines(self, show_list, index, img_shape):

        center = show_list[index][0]
        offset = 30  # 试图偏移的距离

        # 以更高解析度尝试多个方向
        angles = np.linspace(0, 2 * np.pi, num=16, endpoint=False)
        candidates = [
            (int(center[0] + offset * np.cos(angle)), int(center[1] + offset * np.sin(angle)))
            for angle in angles
        ]

        for candidate in candidates:
            if 0 <= candidate[0] < img_shape[1] and 0 <= candidate[1] < img_shape[0]:
                return candidate

        return center  # 如果找不到合适位置，返回原点


    def draw_trajectory(self):

        if len(self.fixation_position) != len(self.duration_list):
            raise ValueError("The length of fixation_position and duration_list should be equal. "
                             "Please check your input.")

        show_list = []

        radius_list = self.radiusList()

        for index, fixation in enumerate(self.fixation_position):
            """
            fixation: (x, y)
            """
            img_orign = self.orig_img.copy()
            img = self.orig_img.copy()
            show_list.append((fixation, self.duration_list[index],index,radius_list[index]))


            if len(show_list) == 1:

                index_ = show_list[0][-2] + 1
                txt = f"[{index_}] {int(show_list[0][1])} ms"

                img = self.draw_dots(img ,show_list[0][-1],fixation[0], fixation[1])

                position = self.find_position_away_from_lines(show_list, index, img.shape)
                img = self.add_text(img, txt, position)

            else:

                for i in range(len(show_list)-1):

                    index_1 = show_list[i][-2]+1
                    txt1 = f"[{index_1}] {int(show_list[i][1])} ms"
                    index_2 = show_list[i+1][-2]+1
                    txt2 = f"[{index_2}] {int(show_list[i+1][1])} ms"
                    center1 = show_list[i][0]
                    center2 = show_list[i+1][0]
                    position1 = self.find_position_away_from_lines(show_list, i, img.shape)
                    position2 = self.find_position_away_from_lines(show_list, i+1, img.shape)
                    img = self.draw_dots(img, show_list[i][-1],center1[0], center1[1])
                    img = self.draw_dots(img, show_list[i+1][-1],center2[0], center2[1])
                    img = self.draw_line(img, center1, center2)
                    img = self.add_text(img, txt1, position1)
                    img = self.add_text(img, txt2, position2)

            if len(show_list) > 2:
                show_list.pop(0)

            # 将透明图层叠加到原始图像上
            alpha = 0.6  # 透明度因子
            cv2.addWeighted(img, alpha, img_orign , 1 - alpha, 0, img_orign )
            # cv2.imshow("img", img_orign)
            self.output.write(img_orign )
            # cv2.waitKey(0)
            # print("saving sub video....")
        #
        # print("###### finish saving all sub videos... #####")

    def radiusList(self):
        """
        Generate a list of radius according to the duration of each fixation.
        """
        min_radiu = self.range_list[0]
        max_radiu = self.range_list[-1]
        radius_list = []
        max_num = max(self.duration_list)
        min_num = min(self.duration_list)
        # 如果 max_num 和 min_num 相等，直接返回相同的半径值
        if max_num == min_num:
            return [min_radiu] * len(self.duration_list)

        duration_num = len(self.duration_list)
        for i in self.duration_list:
            radius = math.ceil((max_radiu - min_radiu) / (max_num - min_num) * (i - min_num) + min_radiu)
            radius_list.append(radius)

        return radius_list


    def draw_dots(self, img, small_radius, x, y):
        """
        Draw a dot on the image at the given position. (fixation)
        """

        small_radius = small_radius
        large_radius = 2 * small_radius

        # 画大圆（红色）
        cv2.circle(img, (int(x), int(y)), large_radius, (0, 0, 255), thickness=-1)

        # 画小圆（黄色）
        cv2.circle(img, (int(x), int(y)), small_radius, (0, 255, 255), thickness=-1)

        return img

    def draw_line(self, img, center1, center2):

        cv2.line(img, (int(center1[0]), int(center1[1])), (int(center2[0]), int(center2[1])), (0, 165, 255), 2)

        return img
