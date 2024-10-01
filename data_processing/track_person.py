import torch
from ultralytics import YOLO

import pickle
from pathlib import Path

import pandas as pd
import sqlite3


def save_pickle(dir_path_str, file_name_str, object_file):
    dir_path = Path(dir_path_str)
    dir_path.mkdir(exist_ok=True)
    file_path = dir_path / f"{file_name_str}.pkl"
    with open(file_path, 'wb') as pickle_file:
        pickle.dump(object_file, pickle_file)
    print(f"saved {file_name_str}")


def init_yolo():
    yolo_path = '/home/liuyu/liuyu_data/models/hf/cv/Ultralytics'
    model = YOLO(f"{yolo_path}/yolov8n.pt")
    return model


def get_track_results(model, video_path_str):
    track_results = model.track(video_path_str, save=True)
    return track_results


def get_video_id_df(db_path_str):
    conn = sqlite3.connect(db_path_str)
    query = "SELECT video_id FROM processing"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_video_path(video_dir_path_str, video_id_str):
    dir_path = Path(video_dir_path_str)
    video_file_path = dir_path / f"{video_id_str}.mp4"
    return video_file_path


def iter_df_and_x(db_path_str, video_path_str, model):
    df = get_video_id_df(db_path_str)
    for index, row in df.iterrows():
        video_path = get_video_path(video_path_str, row['video_id'])
        # print(video_path)
        track_results = get_track_results(model, video_path)
        save_pickle('/home/liuyu/liuyu_data/code/news/data_processing/cv/xinhuashe', row['video_id'], track_results)


def main(db_path_str, video_path_str):
    yolo_model = init_yolo()
    iter_df_and_x(db_path_str, video_path_str, yolo_model)


if __name__ == "__main__":
    db_path_str = r"/home/liuyu/liuyu_data/datasets/paper1/xinhuashe.db"
    video_path_str = r"/home/liuyu/liuyu_data/datasets/paper1/xinhuashe"
    main(db_path_str, video_path_str)
