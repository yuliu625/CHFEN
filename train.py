from trainer import init_model
from untils import load_checkpoint, save_checkpoint

import torch
import torch.nn as nn
import torch.optim as optim

from omegaconf import OmegaConf
from pathlib import Path


def train(config):
    model = init_model(config)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.AdamW(config.model.parameters(), lr=config.train.lr)
    if config['is_from_checkpoint']:
        model, optimizer = load_checkpoint(config, model, optimizer, )


if __name__ == '__main__':
    pass
