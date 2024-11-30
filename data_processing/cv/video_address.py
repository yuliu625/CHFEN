import torch
import cv2
import os
from deep_sort_realtime.deepsort_tracker import DeepSort

import pandas as pd

def process_video(video_file, yolo_model, tracker):
    # video_path = os.path.join(directory, video_file)
    video_path = video_file
    print(f"Processing {video_path}")

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 初始化人物计数
    person_count = {}
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 使用 YOLOv5 进行检测
        results = yolo_model(frame)

        # 筛选出检测到的人员并转换格式
        detections = []
        for *box, conf, cls in results.xyxy[0].tolist():
            if int(cls) == 0:  # 类别 0 是 'person'
                # 从 [x1, y1, x2, y2, confidence] 转换为 [left, top, width, height, confidence]
                x1, y1, x2, y2 = map(float, box)
                width = x2 - x1
                height = y2 - y1
                confidence = float(conf)
                detection = [[x1, y1, width, height], confidence, 'person']
                detections.append(detection)

        # 输出调试信息
        # print("Detections:", detections)

        # 确保 detections 不为空再调用更新函数
        if detections:
            # 更新 DeepSORT 跟踪器
            tracks = tracker.update_tracks(detections, frame=frame)

            # 统计出现次数
            for track in tracks:
                track_id = track.track_id
                if track_id not in person_count:
                    person_count[track_id] = 1
                else:
                    person_count[track_id] += 1

    # 输出结果
    print(f"Results for {video_file}:")

    face_duration_count = 0.0
    for track_id, frames in person_count.items():
        duration = frames / frame_rate
        # print(f'Person {track_id} appeared for {duration:.2f} seconds')
        face_duration_count += duration
    print(f'duration:{face_duration_count}')

    cap.release()    
    return face_duration_count


def main(directory):
    # 加载 YOLOv5 模型
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # 初始化 DeepSORT
    tracker = DeepSort()
    
    # 获取视频文件列表
    # video_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]
    # for video_file in video_files:
    #     process_video(video_file, yolo_model, tracker)

    df = pd.read_csv(f"{directory}/renminribao.csv")
    for index, row in df.iterrows():
        # if row['face_duration'] != 0:
            face_duration_count = process_video(f"{directory}/{row['month']}/{row['video_id']}.mp4", yolo_model, tracker)
            df.loc[index, 'face_duration'] = face_duration_count
            df.to_csv(f"{directory}/renminribao.csv", index=False)


if __name__ == '__main__':
    # 指定视频文件所在的目录
    directory = '/home/liuyu/liuyu_data/datasets/temp/renminribao'
    main(directory)

