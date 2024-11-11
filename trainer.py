from dataset import OriginalMultimodalDataset, MultimodalDataset
from dataset import collate_fn
from model import CHFEN
from untils import load_checkpoint, save_checkpoint, move_batch_to_device

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

import wandb
import os; os.environ["WANDB_MODE"]="offline"

from omegaconf import OmegaConf
from pathlib import Path


def set_wandb(config):
    wandb.init(
        project=config['wandb']['init']['project'],
        name=config['wandb']['init']['name'],
        entity='yu'
    )
    wandb.config = {
        'learning_rate': config['wandb']['config']['learning_rate'],
        'batch_size': config['wandb']['config']['batch_size'],
        'epoch': config['wandb']['config']['epoch'],
        # 'weight_decay': config['wandb']['config']['weight_decay'],
    }


def init_model(config):
    # 实例化并导入模型模型。
    model = CHFEN(config)
    # 冻结参数，这里冻结特征提取编码器的参数。
    # for param in model.total_encoder.parameters():
    #     param.requires_grad = False
    device_str = config['device']
    device = torch.device(device_str)
    model = model.to(device)

    return model


def get_dataloader(train_path_config_path_str, val_path_config_path_str, config=None):
    train_dataloader = DataLoader(
        dataset=MultimodalDataset(
            path_config_path_str=train_path_config_path_str,
            encoder_config_path_str=config['model']['encoder_config_path']
        ),
        batch_size=config['dataloader']['train']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        dataset=MultimodalDataset(
            path_config_path_str=val_path_config_path_str,
            encoder_config_path_str=config['model']['encoder_config_path']
        ),
        batch_size=config['dataloader']['val']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_dataloader, val_dataloader


def train_one_epoch(model, train_dataloader, loss_fn, optimizer, device, config=None):
    # 开始训练。
    model.train()

    total_loss = 0.0
    for data in train_dataloader:
        # 开始
        optimizer.zero_grad()

        # 移动数据。
        data = move_batch_to_device(data, device)
        labels = data['emotion'].to(device)

        # 前向传播
        outputs = model(data)
        loss = loss_fn(outputs, labels)
        # 记录
        total_loss += loss.item()

        # 反向传播
        loss.backward()
        optimizer.step()
    # 记录
    train_avg_loss = total_loss / len(train_dataloader)
    print(f"train_loss: {train_avg_loss}", flush=True)
    # wandb.log({'train_loss': train_avg_loss})


def evaluate_model(model, val_dataloader, loss_fn, device, config=None):
    # 开始评价
    model.eval()

    # 计算测试结果。暂时使用sklearn的工具。
    val_loss = 0.0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for data in val_dataloader:
            # 移动数据。
            data = move_batch_to_device(data, device)
            labels = data['emotion'].to(device)

            # 推理
            outputs = model(data)

            # 测试损失。
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            # 测试准确率。
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_val_loss = val_loss / len(val_dataloader)
    val_accuracy = accuracy_score(all_labels, all_predictions)
    val_f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f"val_loss: {avg_val_loss}", f"val_accuracy: {val_accuracy}", f"val_F1: {val_f1}", flush=True)
    # wandb.log({'val_loss': avg_val_loss, 'val_accuracy': val_accuracy, 'val_F1': val_f1})


def train(config):
    # data相关
    device = torch.device(config['device'])
    train_path_config_path_str = config['dataloader']['train']['dataset_config_path']
    val_path_config_path_str = config['dataloader']['val']['dataset_config_path']
    train_dataloader, val_dataloader = get_dataloader(train_path_config_path_str, val_path_config_path_str, config)

    # 训练相关
    model = init_model(config)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['learning_rate'])
    # 如果需要断点续训
    if config['is_from_checkpoint']:
        model, optimizer = load_checkpoint(config['checkpoint']['path_to_load'], model, optimizer, )

    # 训练主体。
    for epoch in range(config['train']['epoch_start'], config['train']['epoch_end'] + 1):
        print(f"Epoch {epoch}", flush=True)
        train_one_epoch(model, train_dataloader, loss_fn, optimizer, device, config)
        evaluate_model(model, val_dataloader, loss_fn, device, config)
        if epoch % 5 == 0:
            path_to_save = f"{config['checkpoint']['dir_to_save']}/{config['wandb']['init']['name']}_{epoch}.pt"
            save_checkpoint(path_to_save, model, optimizer, )


if __name__ == '__main__':
    config = OmegaConf.load('./config.yaml')
    # set_wandb(config)
    train(config)
