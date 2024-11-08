from model import CHFEN
from dataset import OriginalMultimodalDataset, MultimodalDataset
from torch.utils.data import DataLoader
from dataset import collate_fn

from omegaconf import OmegaConf


if __name__ == '__main__':
    config = OmegaConf.load(r'D:\dcmt\code\dl\paper\news_emotion\config_experiments\baseline_model.yaml')
    # dataset = OriginalMultimodalDataset(path_config_path_str=config['dataloader']['train']['dataset_config_path'])
    dataset = MultimodalDataset(path_config_path_str=config['dataloader']['train']['dataset_config_path'], encoder_config_path_str=r'D:\dcmt\code\dl\paper\news_emotion\config_experiments\encoder.yaml')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    chfen = CHFEN(config)
    for batch in dataloader:
        out = chfen(dataset[0])
        # print(out.shape)
        print(out)
        break


"""
torch.Size([1, 8])
tensor([[-0.0299,  0.0336, -0.0507,  0.0032, -0.0347,  0.0621, -0.0345, -0.0479]],
       grad_fn=<AddmmBackward0>)
"""
