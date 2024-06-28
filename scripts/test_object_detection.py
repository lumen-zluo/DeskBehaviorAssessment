import cv2
from ultralytics import YOLO
import pandas as pd


if __name__ == '__main__':
    cap = cv2.VideoCapture(r"E:\data\subject7\pupil\111725.mp4")

    model_path = '../model/test.pt'

    model = YOLO(model_path)

    eyetracker_gaze_path = r'E:\data\subject7\pupil\111725.mp4_gaze.txt'
    eyetracker_gaze_pd = pd.read_csv(eyetracker_gaze_path, sep=' ', header=None, names=['x', 'y'])
    eyetracker_gaze = eyetracker_gaze_pd.values
    index = 0
    VIDEO_WIDTH = 960
    VIDEO_HEIGHT = 539

    index_to_ruler = 0
    index_to_worksheet = 1
    index_to_eraser = 2
    index_to_pen = 3

    while True:
        _, img = cap.read()

        gaze = eyetracker_gaze[index]

        img = cv2.resize(img, (VIDEO_WIDTH, VIDEO_HEIGHT))

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

        cv2.imshow('img', img)

        index += 1

        cv2.waitKey(1)

