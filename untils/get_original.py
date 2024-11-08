import torch


def get_original_sequence_length(sequence_mask, padding_value=0):
    """输入mask，获得原本序列的长度。"""
    original_sequence_length = sequence_mask.sum(dim=0)
    return original_sequence_length


def get_sequence_length_from_num_faces_vector(sequence: torch.tensor, padding_value=-1):
    """从人脸数量的一维tensor获取原本序列的长度"""
    return (sequence != padding_value).sum().item()


if __name__ == '__main__':
    pass
