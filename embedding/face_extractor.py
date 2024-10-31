from mtcnn import MTCNN

import torch

from PIL import Image

from pathlib import Path
from omegaconf import OmegaConf


class FaceExtractor:
    def __init__(self):

        self.detector = MTCNN()

    def extract_face(self, np_array_image):
        pil_image = self.trans_np_array_to_image(np_array_image)
        faces = self.detector.detect_faces(pil_image)

    def trans_np_array_to_image(self, np_array_image):
        """由于moviepy中的帧图片是ndarray，这里需要转换之后才能输入blip captioner。"""
        pil_img = Image.fromarray(np_array_image)
        return pil_img


if __name__ == '__main__':
    pass
