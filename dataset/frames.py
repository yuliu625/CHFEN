from moviepy.editor import VideoFileClip
import pysrt

from omegaconf import OmegaConf
from pathlib import Path
import math


class Frames:
    """
    输入视频，
    返回采样的已经匹配的图像和字幕。
    """
    def __init__(self, video_id):
        # 导入配置。
        self.path_config = OmegaConf.load('../configs/path.yaml')
        self.base_dir_path = Path(self.path_config['datasets']['base_dir'])

        # 加载视频图像和字幕。
        self.video_path, self.subtitle_path = self.get_video_and_subtitle_path(video_id)
        self.video_clip = self.load_video(self.video_path)
        self.subtitle = self.load_subtitle(self.subtitle_path)

        # 进行采样的序列。
        self.duration = math.floor(self.get_video_info()['duration'])
        self.default_timestamps_list = [i for i in range(self.duration + 1)]

    def get_video_and_subtitle_path(self, video_id):
        """得到视频和字幕的路径。按照dataset文件结构。"""
        video_path = self.base_dir_path / 'video' / f"{video_id}.mp4"
        subtitle_path = self.base_dir_path / 'subtitle' / f"{video_id}.srt"
        return video_path, subtitle_path

    def load_video(self, video_path):
        """使用moviepy处理图像。"""
        return VideoFileClip(str(video_path))

    def load_subtitle(self, subtitle_path):
        """使用pysrt处理字幕。"""
        return pysrt.open(subtitle_path)

    def get_video_info(self):
        """视频基本信息。"""
        return {
            'duration': self.video_clip.duration,
            'fps': self.video_clip.fps,
            'resolution': self.video_clip.size
        }

    def get_frame_image_by_time(self):
        """返回图像序列。类型为原始的numpy.array。"""
        frames_image = []
        for timestamp in self.default_timestamps_list:
            frame = self.video_clip.get_frame(timestamp)
            frames_image.append(frame)

        return frames_image

    def get_frame_subtitle_by_time(self):
        """返回帧的字幕。复杂度可优化。"""
        frames_subtitle = []
        for timestamp in self.default_timestamps_list:
            timestamp = pysrt.SubRipTime(seconds=timestamp)

            found_subtitle = None
            for subtitle in self.subtitle:
                if subtitle.start <= timestamp <= subtitle.end:
                    found_subtitle = subtitle.text
                    break
            frames_subtitle.append(found_subtitle)

        return frames_subtitle


if __name__ == '__main__':
    pass
