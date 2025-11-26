import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json

class IndicHTRDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        with open(label_file, 'r', encoding='utf-8') as f:
            self.annotations = [line.strip().split('\t') for line in f]
        
        self.char_to_idx = self._build_vocab()
        
    def _build_vocab(self):
        chars = set()
        for _, label in self.annotations:
            chars.update(label)
        chars = sorted(list(chars))
        char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}
        char_to_idx['<blank>'] = 0
        return char_to_idx
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_name, label = self.annotations[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        label_encoded = [self.char_to_idx[char] for char in label]
        return image, torch.LongTensor(label_encoded)

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    label_lengths = torch.LongTensor([len(label) for label in labels])
    labels = torch.cat(labels)
    return images, labels, label_lengths
