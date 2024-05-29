import cv2
from ultralytics import YOLO
from OneEuro import EuroPose
import numpy as np
import csv

# 角度组合
keypoint_combinations_with_names = {
    'l_eye-ear-shoulder': (2, 4, 6),
    'r_eye-ear-shoulder': (1, 3, 5),
    'l_ear-shoulder-shoulder': (4, 6, 5),
    'r_ear-shoulder-shoulder': (3, 5, 6),
    'l_hip-shoulder-shoulder': (12, 6, 5),
    'r_hip-shoulder-shoulder': (11, 5, 6),
    'l_elbow-shoulder-hip': (8, 6, 12),
    'r_elbow-shoulder-hip': (7, 5, 11),
    'l_wrist-elbow-shoulder': (6, 8, 10),
    'r_wrist-elbow-shoulder': (5, 7, 9),
    'l_shoulder-hip-hip': (6, 12, 11),
    'r_shoulder-hip-hip': (5, 11, 12),
    'l_knee-hip-hip': (14, 12, 11),
    'r_knee-hip-hip': (13, 11, 12),
    'l_ankle-knee-hip': (16, 14, 12),
    'r_ankle-knee-hip': (15, 13, 11)
}


def calculate_angle(kp1, kp2, kp3):
    """计算由三个点kp1, kp2, kp3组成的角度，其中kp2是顶点"""
    v1 = kp1 - kp2
    v2 = kp3 - kp2
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    angle_rad = np.arccos(dot_product / (norm_v1 * norm_v2))
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def extract_joint(results):
    # 将张量转换为 NumPy 数组
    key_point = np.array(results)

    return key_point, np.where((key_point == [0, 0]).all(axis=1))


def joint_to_list(result):
    joint_data = result[0].keypoints.data[0].to('cpu').numpy()[:, :2].tolist()

    Prepose = []

    for i in joint_data:
        cx, cy = int(i[0]), int(i[1])
        Prepose.append([cx, cy])

    return Prepose


def show_image(keypoints):
    global img
    connections = [[0, 1], [0, 2], [1, 2], [2, 4], [1, 3], [4, 6], [3, 5], [6, 5], [6, 8], [8, 10], [5, 7], [7, 9],
                   [6, 12], [5, 11], [12, 11], [12, 14], [11, 13], [14, 16], [13, 15]]

    # 绘制关节点和连接线
    for point in keypoints:
        x, y = point
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)

    for connection in connections:
        start_point = keypoints[connection[0]]
        end_point = keypoints[connection[1]]
        cv2.line(img, (int(start_point[0]), int(start_point[1])),
                 (int(end_point[0]), int(end_point[1])), (0, 0, 255), 2)


def extract_coordinates(keypoints, index):
    return keypoints.data[0][index].to('cpu').numpy()[:2]


if __name__ == '__main__':
    cap = cv2.VideoCapture(r"F:\backup\Smart-Vest\data\5-51-1\video\5-pre-S1.mp4")
    model_path = '../model/yolov8n-pose.pt'
    model = YOLO(model_path)
    Euro_pose_filter = EuroPose()
    frame_count = 0

    # CSV文件路径
    csv_file_path = r"F:\backup\Smart-Vest\data\5-51-1\video\5-angle-pre-S1.csv"
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', *keypoint_combinations_with_names.keys()])

        while True:
            _, img = cap.read()
            frame_count += 1

            if img is None:
                break

            # 画面裁剪（可选）
            img = img[:, int(1920 * 5 / 13):int(1920 * 10 / 13), :]

            result = model.predict(img)
            PrePose = joint_to_list(result)

            key_point, _ = extract_joint(PrePose)
            Euro_pose_filter.one_euro_pose(key_point, _)
            key_point = Euro_pose_filter.key_point

            angles = {}  # 存储每个角度的计算结果
            for name, combination in keypoint_combinations_with_names.items():
                try:
                    k1, k2, k3 = key_point[combination[0]], key_point[combination[1]], key_point[combination[2]]
                    angle = calculate_angle(k1, k2, k3)
                    angles[name] = f"{angle:.2f}"
                except Exception as e:
                    print(f"Error calculating angle for {name}: {e}")
                    angles[name] = ''

            writer.writerow([frame_count, *angles.values()])
            # 可选：显示图像和角度信息
            # show_image(key_point)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # cv2.imshow('img', img)
            #
            # cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()
