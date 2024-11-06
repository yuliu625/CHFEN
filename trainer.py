from dataset import OriginalMultimodalDataset
from model import CHFEN

import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from pathlib import Path


def init_model(config):
    # 实例化并导入模型模型
    model = CHFEN()
    # if config['is_from_checkpoint']:
    #     model.load_state_dict()

    # 冻结参数，这里冻结特征提取编码器的参数。
    for param in model.total_encoder.parameters():
        param.requires_grad = False

    return model


def get_dataloader(cfg: OmegaConf):
    train_dataloader = DataLoader(dataset=OriginalMultimodalDataset(cfg), batch_size=32, shuffle=True)
    val_dataloader = DataLoader(dataset=OriginalMultimodalDataset(cfg), batch_size=32, shuffle=False)
    return train_dataloader, val_dataloader


def train_model(epoch, model, train_dataloader, loss_fn, optimizer, lr_scheduler, device, cfg: OmegaConf):
    # 开始训练。
    model.train()
    for data in train_dataloader:
        labels = data['emotion'].to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()




def evaluate_model(model, val_dataloader, device, cfg: OmegaConf):
    model.eval()

    with torch.no_grad():
        for data in val_dataloader:
            labels = data['emotion'].to(device)
            outputs = model(data)
            labels = labels.to(device)



if __name__ == '__main__':
    pass
