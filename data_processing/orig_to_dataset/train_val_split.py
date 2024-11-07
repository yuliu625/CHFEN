import pandas as pd
from sklearn.model_selection import train_test_split

from pathlib import Path


def news_emotion_train_test_split(all_data_path_str, dir_to_save_str):
    df = pd.read_json(all_data_path_str, dtype={'video_id': str})

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    dir_to_save = Path(dir_to_save_str)
    train_df.to_json(dir_to_save / 'train_data.json', orient='records', force_ascii=False)
    test_df.to_json(dir_to_save / 'val_data.json', orient='records', force_ascii=False)


if __name__ == '__main__':
    all_data_path_str = r""
    dir_to_save_str = r""
    news_emotion_train_test_split(all_data_path_str, dir_to_save_str)
