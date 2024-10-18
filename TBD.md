# TBD

## 图像处理
### 问题
- 需要处理的图像对象是一系列的图像帧，其中包含temporal information。
- 如何整合一系列的图像信息。
- 输入的序列长度不一致。
- 最终分类器。

### 处理方案
- 简单做法：
  - 时间信息，加入输入的位置编码。
  - 变长序列：
    - 使用padding，因为帧数少。
    - 使用attention pooling。
  - 分类器，linear + softmax。多个模态共用一个分类器。

- two-stream network，计算optical flow。弃用。
- 3D transformer。



## 模态融合
### 问题
- 特征融合
- 决策融合
  - 是否需要设置gate，似乎有gate决定各个模态的权重会更合理。

