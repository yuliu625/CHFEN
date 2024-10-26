# from .title_dataset import
from .frames import Frames

import torch
from torch.utils.data import Dataset

import pandas as pd
from omegaconf import OmegaConf
from pathlib import Path


class EmotionDataset(Dataset):
    """
    加载原始的多模态数据。
    """
    def __init__(self, path_config_path_str='../configs/path.yaml'):
        # 导入配置。
        self.path_config = OmegaConf.load(path_config_path_str)

        # 加载视频、字幕、音频的路径。
        self.base_dir = Path(self.path_config['datasets']['base_dir'])
        self.base_video_dir = Path(self.path_config['datasets']['base_video_dir'])
        self.base_subtitle_dir = Path(self.path_config['datasets']['base_subtitle_dir'])
        self.base_audio_dir = Path(self.path_config['datasets']['base_audio_dir'])

        # 导入主控制文件
        self.all_data = pd.read_json(self.path_config['datasets']['base_all_data'], dtype={'video_id': str})

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return {
            'title': self.all_data.loc[idx, 'title'],
            'emotion': self.all_data.loc[idx, 'emotion'],
        }

    def get_frames_data(self, video_id):
        frames = Frames(video_id)
        video_info = frames.get_video_info()
        frames_image = frames.get_frame_image_by_time()
        frames_subtitle = frames.get_frame_subtitle_by_time()
        return {
            'images': frames_image,
            'subtitles': frames_subtitle
        }


if __name__ == '__main__':
    pass
