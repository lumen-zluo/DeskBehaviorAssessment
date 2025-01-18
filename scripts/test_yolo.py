import ultralytics
import cv2
from ultralytics import YOLO
from OneEuro import EuroPose
import numpy as np



def extract_joint(results):

    # 将张量转换为 NumPy 数组
    key_point = np.array(results)

    return key_point, np.where((key_point == [0, 0]).all(axis=1))

def joint_to_list(result):

    joint_data = result[0].keypoints.data[0].to('cpu').numpy()[:,:2].tolist()

    Prepose = []

    for i in joint_data:

        cx, cy = int(i[0]), int(i[1])
        Prepose.append([cx, cy])

    return Prepose

def show_image(keypoints):

    global  img
    connections = [[0,1],[0,2],[1,2],[2,4],[1,3],[4,6],[3,5],[6,5],[6,8],[8,10],[5,7],[7,9],[6,12],[5,11],[12,11],[12,14],[11,13],[14,16],[13,15]]

    # 绘制关节点和连接线
    for point in keypoints:
        x, y = point
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)

    for connection in connections:
        start_point = keypoints[connection[0]]
        end_point = keypoints[connection[1]]
        cv2.line(img, (int(start_point[0]), int(start_point[1])),
                 (int(end_point[0]), int(end_point[1])), (0, 0, 255), 2)





if __name__ == '__main__':
    cap = cv2.VideoCapture(r"D:\Projects\SmartWriting\video production\materials\raw_skeleton.mp4")

    model_path = '../model/yolov8n-pose.pt'

    model = YOLO(model_path)

    Euro_pose_filter = EuroPose()

    # 获取输入视频的帧率、宽度和高度
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_file = "output_video.mp4"

    # 创建输出视频编码器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    while True:

        _, img = cap.read()

        if not _:
            break

        # img = cv2.imread(r"C:\Users\Owner\Downloads\test.png")

        result = model.predict(img, conf=0.6)

        img = result[0].plot()

        output_video.write(img)

        # PrePose = joint_to_list(result)
        #
        # key_point, result = extract_joint(PrePose)
        #
        # Euro_pose_filter.one_euro_pose(key_point, result)
        #
        # key_point = Euro_pose_filter.key_point
        #
        # print(key_point)
        #
        # # print(key_point)
        # #
        # show_image(key_point)

        # display video
        # img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
        # cv2.imshow('img', img)
        # cv2.waitKey(1)



