from .frames import Frames
from .audio import Audio
from .label_map import label_map

import torch
# import torchvision.transforms as transforms
from torch.utils.data import Dataset

import pandas as pd
from omegaconf import OmegaConf
from pathlib import Path


class OriginalMultimodalDataset(Dataset):
    """
    加载原始的多模态数据。
    """
    def __init__(self, is_need_audio=True, path_config_path_str='../configs/path.yaml'):
        # 导入配置。
        self.path_config = OmegaConf.load(path_config_path_str)

        # 加载视频、字幕、音频的路径。
        self.base_dir = Path(self.path_config['datasets']['base_dir'])
        self.base_video_dir = Path(self.path_config['datasets']['base_video_dir'])
        self.base_subtitle_dir = Path(self.path_config['datasets']['base_subtitle_dir'])
        self.base_audio_dir = Path(self.path_config['datasets']['base_audio_dir'])

        # 导入主控制文件。
        self.is_need_audio = is_need_audio
        self.all_data = pd.read_json(self.path_config['datasets']['base_all_data'], dtype={'video_id': str})

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        frames_data = self.get_frames_data(self.all_data.loc[idx, 'video_id'])
        result = {
            'title': self.all_data.loc[idx, 'title'],
            'emotion_name': self.all_data.loc[idx, 'emotion'],
            'emotion': label_map[self.all_data.loc[idx, 'emotion']],
            'scene': frames_data['images'],
            'subtitle': frames_data['subtitles'],
        }
        if self.is_need_audio:
            audio_data = self.get_audio_data(self.all_data.loc[idx, 'video_id'])
            result = result | audio_data
        return result

    def get_frames_data(self, video_id):
        frames = Frames(video_id)
        video_info = frames.get_video_info()
        frames_image = frames.get_frame_image_by_time()
        frames_subtitle = frames.get_frame_subtitle_by_time()
        return {
            'images': frames_image,
            'subtitles': frames_subtitle
        }

    def get_audio_data(self, video_id):
        audio = Audio(video_id)
        waveform, sample_rate = audio.load_audio()
        return {
            'audio_waveform': waveform,
            'audio_sample_rate': sample_rate
        }


if __name__ == '__main__':
    pass
