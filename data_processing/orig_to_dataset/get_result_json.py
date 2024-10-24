import pandas as pd
import sqlite3


def concat_json():
    df1 = pd.read_json(r'', dtype={"video_id": str})
    df2 = pd.read_json(r'', dtype={"video_id": str})
    df3 = pd.read_json(r'', dtype={"video_id": str})

    combined_df = pd.concat([df1, df2, df3], ignore_index=True)
    combined_df.rename(columns={'video_detail_title': 'title'}, inplace=True)
    print(combined_df)

    combined_df.to_json(r'', orient='records', force_ascii=False)


def load_db(db_path_str):
    return sqlite3.connect(db_path_str)


def load_json(json_path_str):
    return pd.read_json(json_path_str, dtype={"video_id": str})


def get_df(db, json_df, account_name):
    df_title = pd.read_sql("SELECT * FROM detail", con=db)
    df_duration = pd.read_sql("SELECT * FROM processing", con=db)
    result_df =  json_df.merge(df_title, on='video_id', how='inner').merge(df_duration, on='video_id', how='inner')
    result_df = result_df[['video_id', 'video_detail_title', 'duration', 'emotion']]
    result_df['account_name'] = account_name
    return result_df


def main(db_path_str, json_path_str, account_name, path_to_save):
    db = load_db(db_path_str)
    json_df = load_json(json_path_str)
    get_df(db, json_df, account_name).to_json(path_to_save, orient='records', force_ascii=False)


if __name__ == '__main__':
    db_path_str = r""
    json_path_str = r""
    account_name = r""
    path_to_save = r""
    main(db_path_str, json_path_str, account_name, path_to_save)
