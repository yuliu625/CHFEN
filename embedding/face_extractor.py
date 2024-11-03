# from mtcnn import MTCNN  # 使用mtcnn需要tensorflow。
from facenet_pytorch import MTCNN

import torch

from PIL import Image

from pathlib import Path
from omegaconf import OmegaConf


class FaceExtractor:
    def __init__(self):
        """这里使用mtcnn来提取人脸。"""
        self.detector = MTCNN(keep_all=True)  # 需要检测所有的人脸

    def extract_face(self, np_array_image):
        pil_image = self.trans_np_array_to_image(np_array_image)
        img_area = pil_image.width * pil_image.height

        boxes, probs = self.detector.detect(pil_image)

        faces_with_ratios = []
        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)  # 获取人脸边界框并转换为整数

                # 裁剪人脸区域
                face_image = pil_image.crop((x1, y1, x2, y2))

                # 计算人脸面积和占比
                face_area = (x2 - x1) * (y2 - y1)
                face_area_ratio = face_area / img_area

                # 将人脸图像和占比存入列表
                faces_with_ratios.append((face_image, face_area_ratio))  # 这里返回的是pil对象，以及计算的占比。
        return faces_with_ratios

    def trans_np_array_to_image(self, np_array_image):
        """由于moviepy中的帧图片是ndarray，这里需要转换之后才能输入blip captioner。"""
        pil_img = Image.fromarray(np_array_image)
        return pil_img


if __name__ == '__main__':
    pass
