from dataset.frames import Frames
from dataset.audio import Audio
from dataset.label_map import label_map

from embedding import TextEncoder, Captioner, ImageEncoder, FaceExtractor, AudioEncoder

import torch
# import torchvision.transforms as transforms
from torch.utils.data import Dataset

import pandas as pd
from omegaconf import OmegaConf
from pathlib import Path


class MultimodalDataset(Dataset):
    """
    多模态数据。这个类可以被dataloader以batch形式加载。
    数据完全转换成tensor。
    """
    def __init__(self, path_config_path_str='../configs/path.yaml', encoder_config_path_str=r'D:\dcmt\code\dl\paper\news_emotion\config_experiments\encoder.yaml'):
        self.is_need_caption = True
        # 导入配置。
        self.path_config_path_str = path_config_path_str
        self.path_config = OmegaConf.load(path_config_path_str)
        self.encoder_config = OmegaConf.load(encoder_config_path_str)
        self.device_str = self.encoder_config['device']
        self.device = torch.device(self.device_str)

        # 加载视频、字幕、音频的路径。
        self.base_dir = Path(self.path_config['datasets']['base_dir'])
        self.base_video_dir = Path(self.path_config['datasets']['base_video_dir'])
        self.base_subtitle_dir = Path(self.path_config['datasets']['base_subtitle_dir'])
        self.base_audio_dir = Path(self.path_config['datasets']['base_audio_dir'])

        # 导入主控制文件。
        self.all_data = pd.read_json(self.path_config['datasets']['base_all_data'], dtype={'video_id': str})

        # 定义好的一系列的处理和编码器。
        self.text_encoder = TextEncoder(encoder_config_path_str=encoder_config_path_str)
        self.captioner = Captioner(encoder_config_path_str=encoder_config_path_str)
        self.image_encoder = ImageEncoder(encoder_config_path_str=encoder_config_path_str)
        self.face_extractor = FaceExtractor(device_str=self.device_str)
        self.audio_encoder = AudioEncoder(encoder_config_path_str=encoder_config_path_str)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        original_data_dict = self.get_original_data_dict(idx)
        processed_data_dict = self.get_processed_data_dict(original_data_dict)
        return processed_data_dict

    def get_original_data_dict(self, idx):
        frames_data = self.get_frames_data(self.all_data.loc[idx, 'video_id'])
        # 这个类中，音频数据是一定需要的。
        audio_data = self.get_audio_data_dict(self.all_data.loc[idx, 'video_id'])
        result = {
            'title': self.all_data.loc[idx, 'title'],
            'emotion_name': self.all_data.loc[idx, 'emotion'],
            'emotion': torch.tensor(label_map[self.all_data.loc[idx, 'emotion']], dtype=torch.long),
            'scenes': frames_data['images'],  # 这是一个list。
            'subtitles': frames_data['subtitles'],  # 这是一个list。
            'audio_waveform': audio_data['audio_waveform'],
            'audio_sample_rate': audio_data['audio_sample_rate'],
        }
        return result

    def get_frames_data(self, video_id):
        frames = Frames(video_id, path_config_path_str=self.path_config_path_str)
        video_info = frames.get_video_info()
        frames_image = frames.get_frame_image_by_time()
        frames_subtitle = frames.get_frame_subtitle_by_time()
        return {
            'images': frames_image,
            'subtitles': frames_subtitle,
        }

    def get_audio_data_dict(self, video_id):
        audio = Audio(video_id, path_config_path_str=self.path_config_path_str)
        waveform, sample_rate = audio.load_audio()
        return {
            'audio_waveform': waveform,
            'audio_sample_rate': sample_rate
        }

    def get_processed_data_dict(self, data_dict):
        result = {
            'emotion': data_dict['emotion'],
            # 'title': data['title'],
            'title_embedding': self.text_encoder.encode(data_dict['title']),
            'scene_embeddings': self.get_scene_embeddings(data_dict['scenes']),  # shape(seq_length, embedding_dim)
            'num_faces': self.get_face_embeddings(data_dict['scenes'])[0],  # shape(seq_length)
            'face_embeddings': self.get_face_embeddings(data_dict['scenes'])[1],  # shape(seq_length, embedding_dim)
            'text_embeddings': self.get_text_embeddings(data_dict['scenes'], data_dict['subtitles']),  # shape(seq_length, embedding_dim)
            # 'audio': data['audio'],
        }
        audio_embedding = self.get_audio_embedding((data_dict['audio_waveform'], data_dict['audio_sample_rate']))
        result = result | audio_embedding
        return result

    def get_scene_embeddings(self, scenes):
        scene_embedding_list = [self.get_scene_embedding(scene) for scene in scenes]
        return torch.cat(scene_embedding_list, dim=0)

    def get_face_embeddings(self, scenes):
        face_embedding_list = [self.get_face_embedding(scene) for scene in scenes]
        num_faces = torch.tensor([item[0] for item in face_embedding_list]).to(self.device)
        face_embeddings = [item[1] for item in face_embedding_list]
        return num_faces, torch.cat(face_embeddings, dim=0)

    def get_text_embeddings(self, scenes, subtitles):
        text_embeddings_list = []
        for i in range(len(subtitles)):
            scene = scenes[i]
            subtitle = subtitles[i]
            text_embeddings_list.append(self.get_text_embedding(scene, subtitle))

        return torch.cat(text_embeddings_list, dim=0)

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

        if num_faces == 0:
            # 出现一张图像中没有人脸的情况。以非常小的数字代表。
            return 0, torch.randn(1, 768) * 1e-6

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
