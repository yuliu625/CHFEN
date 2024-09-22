import sqlite3
import pandas as pd

from pathlib import Path


def get_df_face_ratio(json_path_str):
    json_path = Path(json_path_str)
    df = pd.read_json(json_path)

    video_id = df[0].str.split('_').str[0].str.split('-').str[1]
    video_id.name = 'video_id'

    face_ratio = df[0].str.split('_').str[-1].str[:4]
    face_ratio.name = 'face_ratio'
    face_ratio = face_ratio.astype(float)

    res_df = pd.concat([video_id, face_ratio], axis=1)
    return res_df


def save_df_to_db(df, db_path_str):
    db_path = Path(db_path_str)
    conn = sqlite3.connect(db_path)
    df.to_sql('face_ratio', conn, if_exists='replace', index=False)
    conn.close()


def main(json_path_str, db_path_str):
    df = get_df_face_ratio(json_path_str)
    save_df_to_db(df, db_path_str)
    # print(df)


if __name__ == '__main__':
    json_path_str = r"./yangshixinwen.json"
    db_path_str = r"./yangshixinwen.db"
    main(json_path_str, db_path_str)
