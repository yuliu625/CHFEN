from .original_dataset import OriginalMultimodalDataset

from embedding import TextEncoder, Captioner, AudioEncoder

import torch
import torchvision.transforms as transforms


class ProcessedMultimodalDataset(OriginalMultimodalDataset):
    def __init__(self, image_transform, is_need_audio=True, path_config_path_str='../configs/path.yaml'):
        super().__init__(is_need_audio, path_config_path_str)
        self.image_transform = image_transform

    def __getitem__(self, idx):
        data = super().__getitem__(idx)


if __name__ == '__main__':
    pass
