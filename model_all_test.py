from model import CHFEN
from dataset import OriginalMultimodalDataset

from omegaconf import OmegaConf


if __name__ == '__main__':
    config = OmegaConf.load(r'D:\dcmt\code\dl\paper\news_emotion\config_experiments\baseline_model.yaml')
    dataset = OriginalMultimodalDataset(path_config_path_str=config['dataloader']['train']['dataset_config_path'])
    chfen = CHFEN(config)
    out = chfen(dataset[0])
    print(out.shape)
    print(out)


"""
torch.Size([1, 8])
tensor([[-0.0299,  0.0336, -0.0507,  0.0032, -0.0347,  0.0621, -0.0345, -0.0479]],
       grad_fn=<AddmmBackward0>)
"""
