import os
import av
import cv2

output_file = 'output_video.avi'
fps = 30
resolution = (960, 538)
# 使用 VideoWriter 对象创建视频编写器
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, fps, resolution)
gaze_file = r"G:\big system 5 version\Keyframe\ori_video\2024-4-13\Cheung Yuen_001\eyetracker\1.txt"
with open(gaze_file, "r") as file:
    gaze = file.readlines()
    gaze_whole_list = [[int(j) for j in i.strip().split(" ")] for i in gaze]
video_path = r"G:\big system 5 version\Keyframe\ori_video\2024-4-13\Cheung Yuen_001\eyetracker\1.mp4"
container = av.open(video_path)
frame_gen = container.decode(video=0)
output_file = 'output_video.avi'
fps = 10
resolution = (960, 538)
# 使用 VideoWriter 对象创建视频编写器
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, fps, resolution)
for i,frame in enumerate(frame_gen):
    img = frame.to_ndarray(format='bgr24')
    x,y = gaze_whole_list[i]
    cv2.circle(img, (int(x), int(y)), 10, (0, 0,255), -1)
    cv2.imshow("Frame",img)
    cv2.waitKey(1)
    out.write(img)