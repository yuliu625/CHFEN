from moviepy.editor import VideoFileClip
import pysrt

from omegaconf import OmegaConf, DictConfig
from pathlib import Path


class Frames:
    def __init__(self, video_id):
        self.path_config = OmegaConf.load('../configs/path.yaml')
        self.base_dir_path = Path(self.path_config['datasets']['base_dir'])

        self.video_path, self.subtitle_path = self.get_video_and_subtitle_path(video_id)
        self.video = self.load_video(frame)

    def get_video_and_subtitle_path(self, video_id):
        video_path = self.base_dir_path / f"{video_id}.mp4"
        subtitle_path = self.base_dir_path / f"{video_id}.srt"
        return video_path, subtitle_path

    def load_video(self, video_path):
        clip = VideoFileClip(video_path)

    def load_subtitle(self, subtitle_path):
        subs = pysrt.open(subtitle_path)

    def get_frame_image_by_time(self, frame):
        pass

    def get_frame_subtitle_by_time(self, frame):
        pass






if __name__ == '__main__':
    pass
