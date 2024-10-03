import torch

import pickle
from pathlib import Path


def open_pickle(file_path_str):
    with open(file_path_str, 'rb') as f:
        return pickle.load(f)


def save_pickle(file_path_str, object_file):
    with open(file_path_str, 'wb') as f:
        pickle.dump(object_file, f)


def get_video_result(frame):
    return {
        'boxes': {
            'cls': frame.boxes.cls,
            'conf': frame.boxes.conf,
            'data': frame.boxes.data,
            'id': frame.boxes.id,
            'is_track': frame.boxes.is_track,
            'orig_shape': frame.boxes.orig_shape,
            'shape': frame.boxes.shape,
            'xywh': frame.boxes.xywh,
            'xywhn': frame.boxes.xywhn,
            'xyxy': frame.boxes.xyxy,
            'xyxyn': frame.boxes.xyxyn,
        },
        'keypoints': frame.keypoints,
        'masks': frame.masks,
        'names': frame.names,
        'obb': frame.obb,
        'orig_img': frame.orig_img,
        'orig_shape': frame.orig_shape,
        'probs': frame.probs,
        'speed': frame.speed,
    }


def check_and_get_person_result(frame_res):
    cls = frame_res['boxes']['cls']
    person_indexs = []
    result = []
    for index in range(len(cls)):
        if cls[index] == 0:
            person_indexs.append(index)
    # print(person_indexs)
    result = get_frame_result(frame_res, person_indexs)
    return result
    
    
def get_frame_result(frame_res, person_indexs):
    result = []
    for index in person_indexs:
        if frame_res['boxes']['id'] is not None:
            result.append({
                'cls': frame_res['boxes']['cls'][index],
                'conf': frame_res['boxes']['conf'][index],
                'data': frame_res['boxes']['data'][index],
                'id': frame_res['boxes']['id'][index],
                # 'is_track': frame_res['boxes']is_track,
                # 'orig_shape': frame_res['boxes']orig_shape,
                # 'shape': frame_res['boxes']shape,
                'xywh': frame_res['boxes']['xywh'][index],
                'xywhn': frame_res['boxes']['xywhn'][index],
                'xyxy': frame_res['boxes']['xyxy'][index],
                'xyxyn': frame_res['boxes']['xyxyn'][index],
            })
    return result


def get_unique_person_id(video_result):
    person_ids = set()
    for frame_result in video_result:
        for person_result in frame_result:
            person_ids.add(person_result['id'].item())
    return person_ids


def get_track_result(result_path):
    video_result_path = result_path
    video_result = open_pickle(video_result_path)
    video_result_list = [get_video_result(frame) for frame in video_result]
    return video_result_list


def process_video_result_list(video_result_list):
    # print(video_result_list)
    person_result_list = [check_and_get_person_result(frame_result) for frame_result in video_result_list]
    # print(person_result_list)
    person_result = get_unique_person_id(person_result_list)
    # print(person_result)
    print(len(person_result))
    return len(person_result)


def main():
    result_path = Path('/home/liuyu/liuyu_data/code/learn/pk_test_save.pkl')
    path_to_save_pkl = Path('/home/liuyu/liuyu_data/code/learn/pk_test_save_list.pkl')
    path_to_save_json = Path('/home/liuyu/liuyu_data/code/learn/pk_test_save_list.json')

    video_result_list = get_track_result(result_path)
    save_pickle(path_to_save_pkl, video_result_list)
    
    person_number = process_video_result_list(video_result_list)
    

if __name__ == '__main__':
    main()

    # print(video_result.video_result_list)
    # print(video_result.person_ids)
    
    
