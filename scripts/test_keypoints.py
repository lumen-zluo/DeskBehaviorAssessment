# import ultralytics
# import cv2
# from ultralytics import YOLO
#
# if __name__ == '__main__':
#     cap = cv2.VideoCapture(r"F:\c16_post\S1-trimmed.mp4")
#
#     model_path = '../model/yolov8x-pose.pt'
#
#     model = YOLO(model_path)
#
#     while True:
#
#         _,img = cap.read()
#         if img is None:
#             break
#
#         img = img[:,int(1920*1/3):int(1920*2/3),:]
#
#         result = model.predict(img)
#
#         img = result[0].plot()
#
#         img = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
#
#         cv2.imshow('img',img)
#
#         cv2.waitKey(1)
#
import cv2
from ultralytics import YOLO
from OneEuro import EuroPose
import numpy as np
import csv


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
    cap = cv2.VideoCapture(r"D:\Projects\SmartWriting\video production\materials\skeleton.mp4")
    model_path = '../model/yolov8n-pose.pt'
    model = YOLO(model_path)
    Euro_pose_filter = EuroPose()

    # 关键点CSV文件路径
    csv_file_path = r"D:\Projects\SmartWriting\video production\materials\skeleton.csv"

    # 打开CSV文件准备写入
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 假设我们有17个关键点，为每个关键点的x和y坐标写入列标题
        headers = ['Frame']
        for i in range(17):  # 假设有17个关键点
            headers.append(f'KeyPoint{i}_X')
            headers.append(f'KeyPoint{i}_Y')
        writer.writerow(headers)

        frame_count = 0
        while True:
            _, img = cap.read()
            if img is None:
                break
            # 画面裁剪（可选）
            img = img[:, int(1920 * 1 / 3):int(1920 * 2 / 3), :]

            result = model.predict(img)
            PrePose = joint_to_list(result)
            key_point, _ = extract_joint(PrePose)
            Euro_pose_filter.one_euro_pose(key_point, _)
            filtered_key_points = Euro_pose_filter.key_point



            # 增加帧计数
            frame_count += 1

            # 准备要写入的数据
            row_data = [frame_count]
            for point in filtered_key_points:
                # 对于每个关键点，添加x和y坐标
                row_data.append(point[0])
                row_data.append(point[1])

            # 将当前帧的关键点坐标写入CSV
            writer.writerow(row_data)

            # 显示图像等（可选）
            # show_image(filtered_key_points)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # cv2.imshow('img', img)
            #
            # cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

