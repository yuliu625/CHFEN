import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from pathlib import Path
from omegaconf import OmegaConf


class AudioEncoder:
    def __init__(self, model='wav2vec2', encoder_config_path_str='../configs/encoder.yaml'):
        # 导入配置。
        self.config = OmegaConf.load(encoder_config_path_str)

        # encoder_path 并选择模型。
        self.encoder_path = Path(self.config['audio'][model]['path'])

        # 加载 Wav2Vec2 预处理器和模型
        self.processor = Wav2Vec2Processor.from_pretrained(self.encoder_path)
        self.model = Wav2Vec2Model.from_pretrained(self.encoder_path)

    # 读取音频文件（假设为 .wav 格式）
    # def load_audio(file_path):
    #     waveform, sample_rate = torchaudio.load(file_path)
    #     return waveform, sample_rate

    # 将音频数据转换为模型的输入
    def encode(self, audio):
        # 加载音频并确保采样率为 16kHz（Wav2Vec2 模型要求）
        # waveform, sample_rate = load_audio(file_path)
        waveform, sample_rate = audio

        waveform = waveform.mean(dim=0)  # 转换为单声道

        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        # 预处理音频数据
        inputs = self.processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)

        # 通过 Wav2Vec2 模型生成embedding
        with torch.no_grad():
            outputs = self.model(inputs.input_values)

        # return outputs
        # 获取最后一层隐藏状态作为音频嵌入
        # embeddings = outputs.extract_features  # 不选择这个，这个是卷积的低层次信息
        # embeddings = outputs.last_hidden_state  # torch.Size([1, length, 768])
        # embeddings = outputs.pooler_output  # 没有这个
        embeddings = outputs.last_hidden_state.max(dim=1).values
        return embeddings


if __name__ == '__main__':
    pass
