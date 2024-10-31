import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    """
    一般的projection，
    需要指定输入和输出。
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)


class AdaptiveProjectionLayer(nn.Module):
    """
    自适应的投影层，适用于有大量embedding_dim需要改变的时候。
    可以在第一次前向传播的时候，自动指定input_dim。
    但是目前的问题是，torch.load会出现问题。或许可以通过在导入前自动进行一次前向传播解决。
    """
    def __init__(self, output_dim):
        super(AdaptiveProjectionLayer, self).__init__()
        self.output_dim = output_dim
        self.projection = None  # 初始时不定义 Linear 层

    def reset_projection(self, input_dim):
        """初始化 projection 层的参数"""
        self.projection = nn.Linear(input_dim, self.output_dim)
        self.projection.reset_parameters()  # 重置参数以保证一致性

    def forward(self, x):
        # 检查 projection 层是否已定义
        if self.projection is None:
            input_dim = x.size(-1)
            self.reset_projection(input_dim)
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
