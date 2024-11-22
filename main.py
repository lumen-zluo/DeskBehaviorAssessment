import cv2
from ultralytics import YOLO
import pandas as pd
import os


def generate_eye_data():
    model_path = '../model/testv3.pt'

    output_path = '../output'

    model = YOLO(model_path)

    index = 0
    VIDEO_WIDTH = 960
    VIDEO_HEIGHT = 539

    index_to_ruler = 0
    index_to_worksheet = 1
    index_to_eraser = 2
    index_to_pen = 3

    subjects = os.listdir(root_path)

    for subject in subjects:
        subject_path = os.path.join(root_path, subject)

        pupil_path = os.path.join(subject_path, 'pupil')
        camera_path = os.path.join(subject_path, r'3camera\front')
        pupil_files = os.listdir(pupil_path)
        # Find pupil video timestamp started at
        pupil_timestamp = 0
        for filename in pupil_files:
            if filename.endswith('.mp4') and not filename.startswith('.'):
                pupil_timestamp = filename.split('.')[0]
                pupil_timestamp = int(pupil_timestamp)
                break

        eyetracker_video_path = os.path.join(pupil_path, f'{pupil_timestamp}.mp4')
        eyetracker_gaze_path = os.path.join(pupil_path, f'{pupil_timestamp}.mp4_gaze.txt')
        eyetracker_gaze_pd = pd.read_csv(eyetracker_gaze_path, sep=' ', header=None, names=['x', 'y'])
        eyetracker_gaze = eyetracker_gaze_pd.values

        output_video_path = os.path.join(output_path, f'{subject}_{pupil_timestamp}.mp4')

        cap = cv2.VideoCapture(eyetracker_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {pupil_timestamp}.mp4")
            exit()
        # 获取视频的帧率和尺寸
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 定义视频编码和输出视频文件
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4编码
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 如果没有更多帧，退出循环

            gaze = eyetracker_gaze[index]

            img = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

            results = model.predict(img)

            result = results[0]

            img = result.plot(boxes=False)

            boxes = result.boxes

            for i, box in enumerate(boxes):
                cords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = cords
                class_id = int(box.cls[0].item())
                if x1 < gaze[0] and y2 > gaze[1]:
                    # show object box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    if class_id == index_to_pen:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, 'pen', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    elif class_id == index_to_ruler:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (33, 37, 43), 2)
                        cv2.putText(img, 'ruler', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (33, 37, 43), 2)
                    elif class_id == index_to_worksheet:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (233, 48, 95), 2)
                        cv2.putText(img, 'worksheet', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (233, 48, 95), 2)
                    elif class_id == index_to_eraser:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (95, 173, 101), 2)
                        cv2.putText(img, 'eraser', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (95, 173, 101), 2)

            cv2.circle(img, (int(gaze[0]), int(gaze[1])), 10, (95, 48, 233), 4)

            # 将处理后的帧写入输出视频
            out.write(img)

            # 可选：显示处理后的帧（按'q'键退出）
            # cv2.imshow('Processed Frame', processed_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    root_path = r'D:\DeskBehaviorData'

    generate_eye_data()

