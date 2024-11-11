import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


def collate_fn(batch):
    """
    用于batch_dataset的填充方法。
    对于sequence，我默认这里最多31个。反正feature module会对list做切片。
    不需要处理的：
        emotion
        title_embedding
        audio_embedding
    需要处理的序列：
        scene_embeddings
        num_faces
        face_embeddings
        text_embeddings
        这3个list的长度是一样的。
    """
    # 不需要pad的，直接cat。
    emotion_embedding_batch_list = [data['emotion'] for data in batch]
    title_embedding_batch_list = [data['title_embedding'] for data in batch]
    audio_embedding_batch_list = [data['audio_embedding'] for data in batch]

    # 需要进行pad的。
    scene_embeddings_batch_list = [data['scene_embeddings'] for data in batch]
    num_faces_batch_list = [data['num_faces'] for data in batch]  # 这个的shape是不一样的，是一维的，不过len相同。
    face_embeddings_batch_list = [data['face_embeddings'] for data in batch]
    text_embeddings_batch_list = [data['text_embeddings'] for data in batch]

    # max_length = max([len(num_faces) for num_faces in num_faces_batch_list])
    # num_faces = [pad_vector(num_faces, max_length=32) for num_faces in num_faces_batch_list]

    padded_num_faces = pad_sequence(num_faces_batch_list, batch_first=True, padding_value=-1)
    padded_scene_embeddings = pad_sequence(scene_embeddings_batch_list, batch_first=True)
    # padded_scene_embeddings = pad_sequence_to_max_length(scene_embeddings_batch_list)
    # scene_mask = padded_scene_embeddings != 0
    padded_face_embeddings = pad_sequence(face_embeddings_batch_list, batch_first=True)
    # padded_face_embeddings = pad_sequence_to_max_length(face_embeddings_batch_list)
    # face_mask = padded_face_embeddings != 0
    padded_text_embeddings = pad_sequence(text_embeddings_batch_list, batch_first=True)
    # padded_text_embeddings = pad_sequence_to_max_length(text_embeddings_batch_list)
    # text_mask = padded_text_embeddings != 0

    return {
        'emotion': torch.stack(emotion_embedding_batch_list),
        'title_embedding': torch.stack(title_embedding_batch_list),
        'audio_embedding': torch.stack(audio_embedding_batch_list),
        # 'num_faces': torch.stack(num_faces),
        'num_faces': padded_num_faces,
        'scene_embeddings': padded_scene_embeddings,
        # 'scene_mask': scene_mask,
        'face_embeddings': padded_face_embeddings,
        # 'face_mask': face_mask,
        'text_embeddings': padded_text_embeddings,
        # 'text_mask': text_mask,
    }


def pad_sequence_to_max_length(sequence_list, max_length=32):
    """
    将一个 list 中的每个张量 pad 至指定的最大长度，支持任意维度的张量填充。

    参数：
    - sequence_list: 一个包含变长 tensor 的 list
    - max_length: 填充后的目标长度

    返回：
    - padded_sequences: 一个张量，包含填充后的序列
    """
    padded_sequences = []

    for item in sequence_list:
        # 获取当前序列的形状
        seq_length = item.size(0)

        # 计算需要填充的长度
        padding_size = max_length - seq_length

        if padding_size > 0:
            # 对于每个维度进行填充（只在序列的第一个维度填充）
            padded_item = F.pad(item, (0, padding_size), value=0)
        else:
            # 如果长度已经大于或等于 max_length，则截断
            padded_item = item[:max_length]

        padded_sequences.append(padded_item)

    # 返回一个堆叠的张量
    return torch.stack(padded_sequences)


def pad_vector(tensor, max_length=32):
    """专为一维tensor使用的填充方法，默认填充至最大32。"""
    padding_length = max_length - tensor.size(0)
    if padding_length > 0:
        padded_tensor = F.pad(tensor, (0, padding_length), value=-1)
    else:
        padded_tensor = tensor
    return padded_tensor


def pad_batch(input_list, pad_value):
    list_mask = []
    for i in range(len(input_list)):
        scene_embedding_list = input_list[i]
        # 先记录这里list的长度。
        list_mask.append(len(scene_embedding_list))
        input_list[i] = pad_list(scene_embedding_list, pad_value)
    return list_mask


def pad_list(input_list, pad_item, length=32):
    if len(input_list) < length:
        return input_list + [pad_item] * (length - len(input_list))


if __name__ == '__main__':
    pass
    # print(pad_list([1, 2, 3], 'a'))
