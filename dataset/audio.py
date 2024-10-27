import torch
import torchaudio

from omegaconf import OmegaConf
from pathlib import Path


class Audio:
    def __init__(self, video_id, path_config_path_str='../configs/path.yaml'):
        # 导入配置。
        self.path_config = OmegaConf.load(path_config_path_str)

        # 加载音频的路径。
        self.base_dir = Path(self.path_config['datasets']['base_dir'])
        self.base_audio_dir = Path(self.path_config['datasets']['base_audio_dir'])

        # 加载音频。
        self.audio_path = self.get_audio_path(video_id)

    def get_audio_path(self, video_id):
        return self.base_audio_dir / f"{video_id}.wav"

    def load_audio(self):
        waveform, sample_rate = torchaudio.load(self.audio_path)
        return waveform, sample_rate


if __name__ == '__main__':
    pass
