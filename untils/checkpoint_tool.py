import torch

from pathlib import Path

"""
这里是强化的对于checkpoint的保存和加载。
默认保存model、optimizer、scheduler。

设计了没有scheduler的情况，但是按照我的trainer的方法，scheduler是一定有的。
"""


def save_checkpoint(path_to_save, model, optimizer, scheduler=None):
    file_path = Path(path_to_save)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, file_path)


def load_checkpoint(path_to_checkpoint, model, optimizer, scheduler=None):
    checkpoint = torch.load(path_to_checkpoint)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
        return model, optimizer, scheduler
    else:
        return model, optimizer


if __name__ == '__main__':
    pass
