import torch
import torch.nn as nn


class AdaptiveProjectionLayer(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.projection = None  # 初始化时不定义 Linear 层

    def forward(self, x):
        # 检查 projection 层是否已定义，如果未定义则根据输入维度动态定义
        if self.projection is None:
            input_dim = x.size(-1)  # 获取输入的最后一个维度
            self.projection = nn.Linear(input_dim, self.output_dim)

        # 将输入投影到指定的输出维度
        return self.projection(x)


if __name__ == '__main__':
    # 用法示例
    batch_size = 32
    num_emb = 10
    embedding_dim = 512  # 假设输入的 embedding 维度是 512
    output_dim = 256

    # 创建 AdaptiveProjectionLayer
    projection_layer = AdaptiveProjectionLayer(output_dim)

    # 输入张量
    input_data = torch.randn(batch_size, num_emb, embedding_dim)  # (batch, num_emb, embedding_dim)
    output_data = projection_layer(input_data)

    print("Output shape:", output_data.shape)  # (batch, num_emb, output_dim)
