from datasets import load_dataset
import torch
from torch.utils.data import Dataset

class SNLI_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        premise = item['premise']
        hypothesis = item['hypothesis']
        label = item['label']
        return premise, hypothesis, label

def get_snli_data():
    dataset = load_dataset("snli")
    train_dataset = SNLI_Dataset(dataset['train'])
    val_dataset = SNLI_Dataset(dataset['validation'])
    test_dataset = SNLI_Dataset(dataset['test'])
    print("len train_dataset:", len(train_dataset))
    print("len val_dataset:", len(val_dataset))
    print("len test_dataset:", len(test_dataset))
    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    get_snli_data()
