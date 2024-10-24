from pathlib import Path


def rename_wav(dir_path_str):
    dir_path = Path(dir_path_str)

    for file in dir_path.iterdir():
        print(file.name)
        print(file.name.replace('_instrumental', ''))
        file.rename(dir_path / file.name.replace('_instrumental', ''))


if __name__ == '__main__':
    rename_wav(r"")
