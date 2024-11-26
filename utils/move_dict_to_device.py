
def move_batch_to_device(batch, device):
    """如果数据是dict，然后需要将里面的tensor移动到gpu，用这个方法。"""
    return {key: value.to(device) for key, value in batch.items()}
