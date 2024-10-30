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
        result = {
            'emotion': data['emotion'],
            'title': data['title'],
            # 'audio': data['audio'],
        }
        if self.is_need_audio:
            audio_data = self.get_audio_embedding(data['audio'])
            result = result | audio_data
        return result

    def transform_image(self):
        """将图片进行处理转换。主要是resize。"""
        pass

    def get_facial_image(self):
        """通过scene得到人脸。"""
        pass

    def get_title_embedding(self, title):
        """获取标题的embedding。"""
        return self.text_encoder.encode(title)

    def get_text_embedding(self, is_need_caption=True):
        """获得text部分的embedding。会输入conditioned_text_encoder。"""

    def generate_caption(self, np_array_image):
        """输入ndarray图片，输出text的caption"""
        return self.captioner.generate(np_array_image)

    def from_image_get_caption_list(self, np_array_image_list):
        """根据本dataset的设计，输入scene list，得到对应的caption list。"""
        return [self.generate_caption(np_array_image) for np_array_image in np_array_image_list]

    def get_image_embedding(self):
        """获取image部分的embedding。会输入conditioned_image_encoder。"""

    def get_face_embedding(self):
        pass

    def get_scene_embedding(self):
        pass

    def get_audio_embedding(self, audio):
        """获取audio部分的embedding。直接输入最终的decision模块。需要判断是否"""
        return self.audio_encoder.encode(audio)

    def build_image_transform(self):
        """默认的自建图片transform pipeline。"""


if __name__ == '__main__':
    pass
