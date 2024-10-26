from moviepy.editor import VideoFileClip
import pysrt

from omegaconf import OmegaConf
from pathlib import Path
import math


class Frames:
    def __init__(self, video_id):
        self.path_config = OmegaConf.load('../configs/path.yaml')
        self.base_dir_path = Path(self.path_config['datasets']['base_dir'])

        self.video_path, self.subtitle_path = self.get_video_and_subtitle_path(video_id)
        self.video_clip = self.load_video(self.video_path)
        self.subtitle = self.load_subtitle(self.subtitle_path)

        self.duration = math.floor(self.get_video_info()['duration'])
        self.default_timestamps_list = [i for i in range(self.duration + 1)]

    def get_video_and_subtitle_path(self, video_id):
        video_path = self.base_dir_path / f"{video_id}.mp4"
        subtitle_path = self.base_dir_path / f"{video_id}.srt"
        return video_path, subtitle_path

    def load_video(self, video_path):
        return VideoFileClip(str(video_path))

    def load_subtitle(self, subtitle_path):
        return pysrt.open(subtitle_path)

    def get_video_info(self):
        return {
            'duration': self.video_clip.duration,
            'fps': self.video_clip.fps,
            'resolution': self.video_clip.size
        }

    def get_frame_image_by_time(self):
        frames_image = []
        for timestamp in self.default_timestamps_list:
            frame = self.video_clip.get_frame(timestamp)
            frames_image.append(frame)

        return frames_image

    def get_frame_subtitle_by_time(self):
        frames_subtitle = []
        timestamps_list = [pysrt.SubRipTime(seconds=timestamp) for timestamp in self.default_timestamps_list]
        for timestamp in timestamps_list:
            subtitle = next((sub.text for sub in self.subtitle if sub.start <= target_time <= sub.end),
                            "No subtitle found at this time")

            # 添加字幕到结果中
            frames_subtitle.append(subtitle)

        return frames_subtitle







if __name__ == '__main__':
    pass
