import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import os



class HeatMap:

    def __init__(self, imgs, save_path, save_name, sigma=25, transparent=0.6, show= False):
        self.imgs = imgs
        self.sigma = sigma
        self.transparent = transparent
        self.fixation_position_list = imgs['fixation_position']
        self.duration_list = imgs['duration_list']
        self.save_path = save_path
        self.show = show
        self.save_name = save_name
        self.draw_heatmap()

    def draw_heatmap(self):

        x_coords = [pos[0] for pos in self.fixation_position_list]
        y_coords = [pos[1] for pos in self.fixation_position_list]

        image_width, image_height = self.imgs['img'].shape[1], self.imgs['img'].shape[0]
        grid_size_x = image_width
        grid_size_y = image_height

        #  创建一个二维直方图，将坐标映射到网格上，使用注视时间作为权重
        heatmap, yedges, xedges = np.histogram2d(y_coords, x_coords,
                                                 bins=[grid_size_y, grid_size_x],
                                                 range=[[0, image_height], [0, image_width]],
                                                 weights=self.duration_list)

        # Step 3: 使用高斯模糊处理，使热力图更平滑
        heatmap = gaussian_filter(heatmap, sigma=self.sigma)

        # Step 4: 归一化热力图到 0-255 的范围，便于转换为颜色图
        max_val = np.max(heatmap)

        # 处理 max_val 为 0 的情况，避免除以 0
        if max_val > 0:
            heatmap = (heatmap / max_val * 255).astype(np.uint8)
        else:
            heatmap = np.zeros_like(heatmap, dtype=np.uint8)

        # Step 5: 将热力图转换为伪彩色图像
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Step 6: 加载背景图片
        background_img = self.imgs['img']   # 替换为你的图片路径
        background_img = cv2.resize(background_img, (image_width, image_height))

        # Step 7: 将热力图叠加到图片上 (0.6 为透明度，你可以根据需求调整)
        overlay = cv2.addWeighted(background_img, self.transparent, heatmap_color, 0.4, 0)

        cv2.imwrite(os.path.join(self.save_path, self.save_name), overlay)

        if self.show:
            cv2.imshow('Fixation Heatmap Overlay', overlay)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
