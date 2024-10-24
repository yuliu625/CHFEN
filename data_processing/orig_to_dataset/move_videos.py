from pathlib import Path


def move_videos(source_dir_str, destination_dir_str):
    source_dir = Path(source_dir_str)
    destination_dir = Path(destination_dir_str)

    destination_dir.mkdir(exist_ok=True)
    for folder in source_dir.iterdir():
        if folder.is_dir():
            for file in folder.iterdir():
                if file.is_file():
                    file.rename(destination_dir / file.name.split('-')[1])
    print('finished moving videos')


if __name__ == '__main__':
    source_dir_str = r""
    destination_dir_str = r""
    move_videos(source_dir_str, destination_dir_str)
