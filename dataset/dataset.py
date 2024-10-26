# from .title_dataset import
from .frames import Frames

import torch
from torch.utils.data import Dataset

import pandas as pd
from omegaconf import OmegaConf, DictConfig


class EmotionDataset(Dataset):
    """
    加载原始的多模态数据。
    """
    def __init__(self,):
        # 导入配置。
        self.path_config = OmegaConf.load('../configs/path.yaml')
        self.all_data = pd.read_json(self.path_config['datasets']['base_all_data'], dtype={'video_id': str})

        # 加载视频字幕，音频的路径
        self.base_video_dir = self.path_config['datasets']['base_video_dir']
        self.base_subtitle_dir = self.path_config['datasets']['base_subtitle_dir']
        self.base_audio_dir = self.path_config['datasets']['base_audio_dir']

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        pass

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
