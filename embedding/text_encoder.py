import torch
import torchvision.transforms as transforms

from transformers import RobertaTokenizer, RobertaModel
from transformers import XLMRobertaModel, XLMRobertaTokenizer

from pathlib import Path
from omegaconf import OmegaConf


class TextEncoder:
    """
    一般的文本编码器，将文本编码为embedding。
    这里因为主体为中文，以及caption生成为英文，使用多语言的xlm更好。
    """
    def __init__(self, model='xlm-roberta', encoder_config_path_str='../configs/encoder.yaml'):
        # 导入配置。
        self.config = OmegaConf.load(encoder_config_path_str)

        # encoder_path 并选择模型。
        self.encoder_path = Path(self.config['text'][model]['path'])
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.encoder_path)
        self.model = XLMRobertaModel.from_pretrained(self.encoder_path)

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state  # torch.Size([1, 128, 768])必然被截断或填充到128。
            # embedding = outputs.last_hidden_state.mean(dim=1)
            # embeddings = outputs.pooler_output  # CLS embedding经过一定处理，torch.Size([1, 768])。

        # return outputs
        return embeddings


if __name__ == '__main__':
    text_encoder = TextEncoder()
    print(text_encoder.config)
