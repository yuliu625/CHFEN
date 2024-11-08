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

    num_faces = torch.cat([pad_vector(num_faces) for num_faces in num_faces_batch_list])
    padded_scene_embeddings = pad_sequence(scene_embeddings_batch_list, batch_first=True)
    scene_mask = padded_scene_embeddings != 0
    padded_face_embeddings = pad_sequence(face_embeddings_batch_list, batch_first=True)
    face_mask = padded_face_embeddings != 0
    padded_text_embeddings = pad_sequence(text_embeddings_batch_list, batch_first=True)
    text_mask = padded_text_embeddings != 0

    return {
        'emotion': emotion_embedding_batch_list,
        'title_embedding': torch.cat(title_embedding_batch_list),
        'audio_embedding': torch.cat(audio_embedding_batch_list),
        'num_faces': num_faces,
        'scene_embeddings': scene_embeddings_batch_list,
        'scene_mask': scene_mask,
        'face_embeddings': face_embeddings_batch_list,
        'face_mask': face_mask,
        'text_embeddings': text_embeddings_batch_list,
        'text_mask': text_mask,
    }


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
