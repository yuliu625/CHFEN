from pathlib import Path
import json


def find_mp4_files(folder_path, save_path):
    data = []
    path = Path(folder_path)
    for mp4_file in path.rglob('*.mp4'):
        # print(mp4_file.name)
        data.append(mp4_file.name)
    print(len(data))
    with open(save_path, 'w') as fp:
        json.dump(data, fp)


if __name__ == '__main__':
    folder_path = r"D:\dcmt\dataset\douyin\renminribao\renminribao"
    save_path = r"data.json"
    find_mp4_files(folder_path, save_path)

