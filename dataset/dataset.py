# from .title_dataset import
from .frame_dataset import Frames

import torch
from torch.utils.data import Dataset

import pandas as pd
from omegaconf import OmegaConf, DictConfig


class EmotionDataset(Dataset):
    def __init__(self,):
        self.path_config = OmegaConf.load('../configs/path.yaml')
        self.all_data = pd.read_json(self.path_config['datasets']['base_all_data'], dtype={'video_id': str})

        self.base_video_dir = self.path_config['datasets']['base_video_dir']
        self.base_subtitle_dir = self.path_config['datasets']['base_subtitle_dir']
        self.base_audio_dir = self.path_config['datasets']['base_audio_dir']

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        pass

    def get_frames_data(self, video_id):
        frames = Frames()
        pass


if __name__ == '__main__':
    pass
