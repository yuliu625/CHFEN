from .text_encoder import TextEncoder
from .image_to_text import Captioner
from .image_encoder import ImageEncoder
from .face_extractor import FaceExtractor
# from .facial_encoder import FacialEncoder
from .audio_encoder import AudioEncoder

import torch
import torch.nn as nn

from pathlib import Path
from omegaconf import OmegaConf


class TotalEncoder(nn.Module):
    """
    整合的模块。
    将所有的dataset的输入encode成需要的各种embedding。
    """
    def __init__(self, is_need_caption=True, is_need_audio=True,):
        super().__init__()
        self.is_need_caption = is_need_caption
        self.is_need_audio = is_need_audio

        # 定义好的一系列的处理和编码器。
        self.text_encoder = TextEncoder()
        self.captioner = Captioner()
        self.image_encoder = ImageEncoder()
        self.face_extractor = FaceExtractor()
        self.audio_encoder = AudioEncoder()

    def forward(self, data):
        with torch.no_grad():
            result = {
                'emotion': data['emotion'],
                'title_embedding': self.text_encoder.encode(data['title']),
                'scene_embedding_list': self.get_scene_embedding_list(data['scenes']),
                'face_embedding_list': self.get_face_embedding_list(data['scenes']),
                'text_embedding_list': self.get_text_embedding_list(data['scenes'], data['subtitles']),
                # 'audio': data['audio'],
            }
            if self.is_need_audio:
                audio_embedding = self.get_audio_embedding((data['audio_waveform'], data['audio_sample_rate']))
                result = result | audio_embedding
        return result

    def get_scene_embedding_list(self, scenes):
        return [self.get_scene_embedding(scene) for scene in scenes]

    def get_face_embedding_list(self, scenes):
        return [self.get_face_embedding(scene) for scene in scenes]

    def get_text_embedding_list(self, scenes, subtitles):
        text_embeddings_list = []
        for i in range(len(subtitles)):
            scene = scenes[i]
            subtitle = subtitles[i]
            text_embeddings_list.append(self.get_text_embedding(scene, subtitle))

        return text_embeddings_list

    def get_title_embedding(self, title):
        """获取标题的embedding。"""
        return self.text_encoder.encode(title)

    def get_text_embedding(self, scene, subtitle, is_need_caption=True):
        """获得text部分的embedding。会输入conditioned_text_encoder。"""
        caption = self.captioner.generate(scene)
        result = ''
        if self.is_need_caption:
            # 这里对于text的部分选择的方法是拼接。
            result = subtitle + '\n' + caption
        else:
            result = subtitle
        text_embedding = self.text_encoder.encode(result)
        return text_embedding

    def generate_caption(self, np_array_image):
        """输入ndarray图片，输出text的caption"""
        return self.captioner.generate(np_array_image)

    # def from_image_get_caption_list(self, np_array_image_list):
    #     """根据本dataset的设计，输入scene list，得到对应的caption list。"""
    #     return [self.generate_caption(np_array_image) for np_array_image in np_array_image_list]

    # def get_image_embedding(self):
    #     """获取image部分的embedding。会输入conditioned_image_encoder。"""

    def get_faces_and_ratios_list(self, np_array_image):
        """
        输入np_array的图片，返回一个元组的list，分别是(face_pil_image,face_area_ration)。
        这里默认输入的是scene。
        """
        faces_with_ratios_list = self.face_extractor.extract_face(np_array_image)
        return faces_with_ratios_list

    def get_face_embedding(self, scene, is_need_norm=False):
        """聚合多张脸的语义信息。返回结果是(num_faces,face_embedding)"""
        faces_with_ratios_list = self.get_faces_and_ratios_list(scene)
        num_faces = len(faces_with_ratios_list)

        total_ratio = sum(face_area_ratio for _, face_area_ratio in faces_with_ratios_list)
        weighted_embeddings = []
        for face_image, face_area_ratio in faces_with_ratios_list:
            # 先将脸部的图片进行编码。
            face_embedding = self.image_encoder.encode(face_image)
            if is_need_norm:
                # 这里如果需要进行归一化，就调整原本脸的占比的数值。
                face_area_ratio = face_area_ratio / total_ratio
            weighted_embedding = face_embedding * face_area_ratio
            weighted_embeddings.append(weighted_embedding)

        face_embedding = torch.sum(torch.stack(weighted_embeddings), dim=0)

        return num_faces, face_embedding

    def get_scene_embedding(self, scene):
        return self.image_encoder.encode(scene)

    def get_audio_embedding(self, audio):
        """获取audio部分的embedding。直接输入最终的decision模块。需要判断是否"""
        return {'audio_embedding': self.audio_encoder.encode(audio)}


if __name__ == '__main__':
    pass
