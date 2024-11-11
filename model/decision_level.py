from model.type_encoding import TypeEncoding

import torch
import torch.nn as nn

from pathlib import Path
from omegaconf import OmegaConf


class DecisionModule(nn.Module):
    """
    输入：
        feature-level已经处理过的各个模态的embedding。
    处理：
        给各个模态的embedding加入各自类别的embedding。
        使用self attention加ffn提取信息。
        使用pooling聚合语义信息。
        使用classifier进行分类。
    输出：
        classifier的输出，即softmax的输出。
    """
    def __init__(self, is_need_audio=True, embedding_dim=768, num_heads=8):
        super().__init__()
        self.is_need_audio = is_need_audio

        # 各种embedding的类型编码，需要扩展。
        self.title_type_encoding = TypeEncoding()
        self.image_type_encoding = TypeEncoding()
        self.text_type_encoding = TypeEncoding()
        self.audio_type_encoding = TypeEncoding()

        # self-attention提取特征。
        self.self_attention_layer_1 = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        # self.linear_layer = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.ffn_layer_1 = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
            nn.ReLU(),
        )
        self.self_attention_layer_2 = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        self.ffn_layer_2 = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
            nn.ReLU(),
        )

        # 注意力池化
        self.learnable_pooling_query = nn.Parameter(torch.randn(1, embedding_dim))
        self.attention_pooling_layer = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)

    def forward(self, embedding_dict):
        # 这是上一次传入的的embedding。
        title_embedding = embedding_dict['title_embedding']
        image_embedding = embedding_dict['image_embedding_sequence']
        text_embedding = embedding_dict['text_embedding_sequence']
        audio_embedding = embedding_dict['audio_embedding']

        # 进行类别编码。其实tittle和audio不需要expand。
        title_embedding += self.title_type_encoding().expand_as(title_embedding)
        image_embedding += self.image_type_encoding().expand_as(image_embedding)
        text_embedding += self.text_type_encoding().expand_as(text_embedding)
        audio_embedding += self.audio_type_encoding().expand_as(audio_embedding)

        embeddings = torch.cat((title_embedding, image_embedding, text_embedding, audio_embedding), dim=1)
        # print(embeddings.shape)  # torch.Size([1, 18, 768])

        # 前向传播算法部分
        out, _ = self.self_attention_layer_1(embeddings, embeddings, embeddings)  # torch.Size([1, 18, 768])
        # print(out.shape)
        out = self.ffn_layer_1(out)  # torch.Size([1, 18, 768])
        # print(out.shape)
        out, _ = self.self_attention_layer_2(out, out, out)
        out = self.ffn_layer_2(out)

        # 在执行查询前，需要扩展至batch的shape，即(batch_size, 1, embedding_dim)
        query = self.learnable_pooling_query.unsqueeze(0).expand(out.shape[0], -1, -1)  # torch.Size([2, 1, 768])
        # print(query.shape)
        out, _ = self.attention_pooling_layer(query, out, out)  # torch.Size([2, 1, 768])
        # print(out.shape)
        out = out.squeeze(1)
        return out


if __name__ == '__main__':
    pass
