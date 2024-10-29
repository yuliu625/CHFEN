from .original_dataset import OriginalMultimodalDataset

from embedding import TextEncoder, Captioner, AudioEncoder

import torch
import torchvision.transforms as transforms


class ProcessedMultimodalDataset(OriginalMultimodalDataset):
    def __init__(self, is_need_audio=True, path_config_path_str='../configs/path.yaml'):
        super().__init__(is_need_audio, path_config_path_str)
        # self.image_transform = image_transform

        self.text_encoder = TextEncoder()
        self.captioner = Captioner()
        self.audio_encoder = AudioEncoder()

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return data

    def transform_image(self):
        """将图片进行处理转换。主要是resize。"""
        pass

    def get_facial_image(self):
        """通过scene得到人脸。"""
        pass

    def get_title_embedding(self):
        """获取标题的embedding。"""

    def get_text_embedding(self, is_need_caption=True):
        """获得text部分的embedding。会输入conditioned_text_encoder。"""

    def get_image_embedding(self):
        """获取image部分的embedding。会输入conditioned_image_encoder。"""

    def get_face_embedding(self):
        pass

    def get_scene_embedding(self):
        pass

    def get_audio_embedding(self):
        """获取audio部分的embedding。直接输入最终的decision模块。需要判断是否"""

    def build_image_transform(self):
        """默认的自建图片transform pipeline。"""


if __name__ == '__main__':
    pass
