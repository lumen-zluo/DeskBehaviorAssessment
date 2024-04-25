import cv2
from ultralytics import YOLO


if __name__ == '__main__':
    cap = cv2.VideoCapture(r"E:\data\subject4\pupil\143035.mp4")

    model_path = '../model/yolov8x.pt'

    model = YOLO(model_path)

    while True:
        _, img = cap.read()

        results = model.predict(img)

        img = results[0].plot(boxes=True)

        # img = cv2.resize(results, (int(img.shape[1] / 2), int(img.shape[0] / 2)))

        cv2.imshow('img', img)

        cv2.waitKey(1)
