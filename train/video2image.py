import cv2
import os


def video2imgs(videoPath, imgPath):
    global count
    if not os.path.exists(imgPath):
        os.makedirs(imgPath)  # 目标文件夹不存在，则创建

    for filename in os.listdir(videoPath):
        if filename.endswith('.mp4'):
            video = os.path.join(videoPath, filename)
            cap = cv2.VideoCapture(video)  # 获取视频
            judge = cap.isOpened()  # 判断是否能打开成功
            print(judge)
            fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率，视频每秒展示多少张图片
            print('fps:', fps)

            frames = 1  # 用于统计所有帧数

            while judge:
                flag, frame = cap.read()  # 读取每一张图片 flag表示是否读取成功，frame是图片
                if not flag:
                    print(flag)
                    print("Process finished!")
                    break
                else:
                    if frames % 20 == 0:  # 每隔20帧抽一张
                        imgname = 'jpgs_' + str(count).rjust(3, '0') + ".jpg"
                        newPath = os.path.join(imgPath, imgname)
                        # print(imgname)
                        cv2.imwrite(newPath, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                        # cv2.imencode('.jpg', frame)[1].tofile(newPath)
                        count += 1
                frames += 1
            cap.release()
            print("共有 %d 张图片" % (count - 1))



if __name__ == '__main__':
    root_path = r'D:\Code\DesktopData'
    output_path = r'D:\output'
    count = 1379
    for name in os.listdir(root_path):
        path = os.path.join(root_path, name, 'pupil')
        if os.path.isdir(path):
            print("正在處理子目录:", path)
            video2imgs(path, output_path)
