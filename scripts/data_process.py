from ultralytics import YOLO
import cv2
import time
import os
from util.common import check_file_exists
import xml.etree.ElementTree as ET
import data_segment


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame = 0
    directory = os.path.dirname(video_path)
    joint_path = os.path.join(directory, 'joint.xml')
    root = ET.Element("root")
    time_path = os.path.join(directory, 'realsense_time.txt')

    # Read eyetracker time data
    eyetracker_time = []
    with open(time_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            eyetracker_time.append(line)

    # Writ data into xml
    while cap.isOpened():
        _, img = cap.read()
        if _:
            start = time.time()
            result = model.predict(img, conf=0.5)
            image = result[0].plot()

            image = cv2.resize(image, (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5)))
            cv2.imshow("img", image)
            cv2.waitKey(1)

            key_points = result[0].keypoints.cpu().numpy()

            if key_points.has_visible:
                for index, value in enumerate(key_points.data):
                    keypoint_element = ET.SubElement(root, "keypoint")
                    time_element = ET.SubElement(keypoint_element, "timestamp")
                    time_element.text = eyetracker_time[index]

                    for item in value:
                        if item[2] >= 0.5:
                            # confidence >= 0.5
                            position_element = ET.SubElement(keypoint_element, "position")
                            x_element = ET.SubElement(position_element, "x")
                            x_element.text = str(item[0])
                            y_element = ET.SubElement(position_element, "y")
                            y_element.text = str(item[1])
                            confidence_element = ET.SubElement(position_element, "confidence")
                            confidence_element.text = str(item[2])

            # print("key points:", key_points)
            end = time.time()
            # print("use time: ", end - start)

            # line = str(int(txt_x)) + " " + str(int(txt_y)) + "\n"
            # file.write(line)

            frame = frame + 1
            # xml_string = ET.tostring(root, encoding="utf-8")

        else:
            tree = ET.ElementTree(root)
            tree.write(joint_path, encoding="utf-8", xml_declaration=True)
            print('Process finished with {} frames'.format(frame))
            break


if __name__ == '__main__':

    date_directories = os.listdir('../data')
    model = YOLO('../model/yolov8x-pose.pt')

    for date_directory in date_directories:
        date_path = os.path.join('../data', date_directory)
        if os.path.isdir(date_path):
            user_directories = os.listdir(date_path)
            for user_directory in user_directories:
                user_path = os.path.join(date_path, user_directory)
                realsense_video_path = os.path.join(user_path, 'realsense/video.mp4')
                realsense_joint_path = os.path.join(user_path, 'realsense/joint.csv')
                handwriting_path = os.path.join(user_path, 'handwriting/handwriting_time.txt')
                eyetracker_path = os.path.join(user_path, 'eyetracker/time.txt')
                synchronize_path = os.path.join(user_path, 'synchronize/data.xml')
                if (check_file_exists(realsense_joint_path) and
                        check_file_exists(handwriting_path) and
                        check_file_exists(eyetracker_path) and
                        not check_file_exists(synchronize_path)):
                    # process_video(realsense_video_path)
                    segment = data_segment.DataSegment(user_path)
                    segment.run_seg()
        else:
            print("{} is not a directory.".format(date_path))
