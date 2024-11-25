import cv2
import numpy as np


def variance_of_laplacian(image ,threshold=100 ):
    # 读取图像
    if image is None:
        print("无法读取图像，请检查路径是否正确。")
        return False

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用拉普拉斯算子
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # 计算方差
    variance = laplacian.var()

    # print(f"拉普拉斯方差: {variance}")

    # 根据方差值判断
    threshold = threshold  # 阈值需要根据具体情况调整

    if variance < threshold:
        # print("图像模糊")
        return False
    else:
        # print("图像清晰")
        return True


def gradient_based_blur_detection(image,threshold = 40 ):

    if image is None:
        print("无法读取图像，请检查路径是否正确。")
        return False

    # 计算Sobel梯度
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)

    # 计算平均梯度幅值
    mean_gradient = np.mean(gradient_magnitude)

    # print(f"平均梯度幅值: {mean_gradient}")

    # 根据平均梯度幅值判断
    threshold = threshold  # 阈值需要根据具体情况调整
    if mean_gradient < threshold:
        # print("图像模糊")
        return False
    else:
        # print("图像清晰")
        return True


if __name__ == '__main__':
    video_path = r"G:\big system 5 version\Keyframe\ori_video\2024-3-26\Cheung Yuen_002\eyetracker\1.mp4"
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        print("##############")
        _,img = cap.read()
        VOL = variance_of_laplacian(img)
        GBD = gradient_based_blur_detection(img)
        if VOL or GBD:

            cv2.putText(img, "True", (100,100),4,3,(0,0,255),2)
            print("True")
        else:

            cv2.putText(img, "False", (100, 100), 4, 3, (0, 0, 255), 2)
            print("False")
        cv2.imshow('img',img)
        cv2.waitKey(1)
