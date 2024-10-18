# TBD

## 图像处理
### 问题
- 需要处理的图像对象是一系列的图像帧，其中包含temporal information。
- 如何整合一系列的图像信息。
- 输入的序列长度不一致。

### 处理方案
- 简单做法：
  - 时间信息，加入输入的位置编码。
  - 变长序列：
    - 使用padding，因为帧数少。
    - 使用attention pooling。
- two-stream networ，计算optical flow。放弃
- 3D



## 模态融合

