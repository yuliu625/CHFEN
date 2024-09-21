import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification

import pandas as pd
import json
from pathlib import Path


def init_model(model_path, num_labels):
    model_path = r"/home/liuyu/liuyu_data/models/hf/nlp/roberta/roberta-large"
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=15)
    return tokenizer, model


class TextDataset(Dataset):
    def __init__(self, file_path, label_map, tokenizer):
        self.tokenizer = tokenizer
        self.data = self.get_data(file_path)
        self.labels = self.data['label']
        self.sentences = self.data['sentence']
        self.label_map = {
            '100': 0,
            '101': 1,
            '102': 2,
            '103': 3,
            '104': 4,
            '106': 5,
            '107': 6,
            '108': 7,
            '109': 8,
            '110': 9,
            '112': 10,
            '113': 11,
            '114': 12,
            '115': 13,
            '116': 14,
        }

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sentence = self.sentences[index]
        label = self.labels[index]
        encoding = self.tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.label_map[label], dtype=torch.long)
        }

    def get_data(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        return pd.DataFrame(data)


def get_dataloader(dataset_path, tokenizer, batch_size):
    train_dataset = TextDataset(Path(r'/home/liuyu/liuyu_data/code/news/data_processing/text/data/my/classification/train.json'), None, tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    return train_dataloader


def train_model(epoch, model, train_dataloader, val_dataloader, loss_fn, optimizer, learning_rate):
    model.train()
    for batch in train_dataloader:
        inputs = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        
        optimizer.zero_grad()
        outputs = model(inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss.mean()
        # loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    print(loss)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'roberta_checkpoint{epoch}.pt')


def evaluate_model(model, test_dataloader):
    with torch.no_grad():
        pass


def save_checkpoint(epoch, model, optimizer, loss, lr_scheduler, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'lr_scheduler_state_dict': lr_scheduler.state_dict()
    }, save_path)


def main():
    tokenizer, model = init_model('', 15)
    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model)
    train_dataloader = get_dataloader('', tokenizer, 32)
    
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    
    for epoch in range(20):
        train_model(epoch, model, train_dataloader, None, loss_fn, optimizer, 1e-6)


if __name__ == "__main__":
    main()
