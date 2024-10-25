import pandas as pd

from omegaconf import OmegaConf
from pathlib import Path


def get_all_data(all_data_path):
    return pd.read_json(all_data_path, dtype={'video_id': str})


def check_files(all_data, file_dir, file_type):
    for video_id in all_data['video_id']:
        path_to_check = file_dir / f'{video_id}{file_type}'
        # print(path_to_check)
        if not path_to_check.exists():
            print(path_to_check)


if __name__ == '__main__':
    path_config = OmegaConf.load('../../configs/path.yaml')
    base_dir = Path(path_config['datasets']['base_dir'])
    video_dir = base_dir / 'video'
    subtitle_dir = base_dir / 'subtitle'

    all_data_path = base_dir / 'all_data.json'
    all_data = get_all_data(all_data_path)

    check_files(all_data, video_dir, '.mp4')
    check_files(all_data, subtitle_dir, '.srt')
