import pandas as pd
import sqlite3
from pathlib import Path


def rename_df(df):
    df['video_id'] = df['video_id'].str.split('-').str[1]
    df.rename(columns={
        'video_url': 'video_detail_url',
        'video_img': 'video_img_url',
        'video_url_': 'video_url_list',
    }, inplace=True)
    df.rename(columns={
        '时长': 'video_detail_duration',
        '标题': 'video_detail_title',
        '点赞': 'video_detail_like',
        '评论': 'video_detail_reply',
        '收藏': 'video_detail_collect',
        '转发': 'video_detail_repost',
        '发布时间': 'video_detail_time',
        '采集时间': 'video_detail_get_time',
    }, inplace=True)


def get_dfs(df):
    df_url = df[['video_id', 'video_detail_url', 'video_img_url', 'video_url_list', 'video_url_0', 'video_url_1', 'video_url_2', ]]
    df_info = df[['video_id', 'video_tittle', 'video_like', 'video_innertext', ]]
    df_time = df[['video_id', 'year', 'month', 'video_detail_time', 'video_detail_get_time', ]]
    df_detail = df[['video_id', 'video_detail_title', 'video_detail_like', 'video_detail_reply', 'video_detail_collect', 'video_detail_repost', 'video_detail_duration', ]]
    df_processing = df[['video_id', 'duration', ]]
    return df_url, df_info, df_time, df_detail, df_processing


def df_to_sql(conn, df_url, df_info, df_time, df_detail, df_processing):
    df_url.to_sql('url', conn, if_exists='replace', index=False)
    df_info.to_sql('info', conn, if_exists='replace', index=False)
    df_time.to_sql('time', conn, if_exists='replace', index=False)
    df_detail.to_sql('detail', conn, if_exists='replace', index=False)
    df_processing.to_sql('processing', conn, if_exists='replace', index=False)


def main(folder_path_str, account_name):
    csv_path = Path(folder_path_str) / f"{account_name}.csv"
    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(f"{account_name}.db")
    df.to_sql(f"{account_name}", conn, if_exists='replace', index=False)
    rename_df(df)
    # create_table()
    df_url, df_info, df_time, df_detail, df_processing = get_dfs(df)
    df_to_sql(conn, df_url, df_info, df_time, df_detail, df_processing)
    conn.close()


if __name__ == '__main__':
    folder_path_str = r"/home/liuyu/liuyu_data/datasets/my/douyin/yangshixinwen"
    account_name = "yangshixinwen"
    main(folder_path_str, account_name)
