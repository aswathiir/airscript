import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from htr_model import HybridHTRModel
from data_loader import IndicHTRDataset, collate_fn
from tqdm import tqdm
import os

def train_multilingual_model(
    language_configs,
    num_epochs=50,
    batch_size=32,
    learning_rate=0.001,
    save_dir='../models/htr'
):
    os.makedirs(save_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load all language datasets
    datasets = []
    global_char_to_idx = {'<blank>': 0}
    
    for lang_name, data_dir, label_file in language_configs:
        dataset = IndicHTRDataset(data_dir, label_file, transform=transform)
        
        # Merge vocabularies
        for char, idx in dataset.char_to_idx.items():
            if char not in global_char_to_idx:
                global_char_to_idx[char] = len(global_char_to_idx)
        
        datasets.append(dataset)
        print(f'{lang_name}: {len(dataset)} samples')
    
    # Combine datasets
    combined_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    num_classes = len(global_char_to_idx)
    print(f'Total vocabulary size: {num_classes}')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')
    
    model = HybridHTRModel(num_classes=num_classes).to(device)
    
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels, label_lengths in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            log_probs = nn.functional.log_softmax(outputs, dim=2)
            
            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=log_probs.size(1),
                dtype=torch.long
            )
            
            loss = ctc_loss(
                log_probs.permute(1, 0, 2),
                labels,
                input_lengths,
                label_lengths
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1} - Average Loss: {avg_loss:.4f}')
        
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'char_to_idx': global_char_to_idx
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
    
    final_path = os.path.join(save_dir, 'checkpoint_finetuned.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': global_char_to_idx
    }, final_path)
    print(f'Final model saved: {final_path}')

if __name__ == '__main__':
    configs = [
        ('Hindi', '../data/raw/hindi_processed/images', '../data/raw/hindi_processed/labels.txt'),
        ('Malayalam', '../data/raw/malayalam_processed/images', '../data/raw/malayalam_processed/labels.txt'),
        ('Tamil', '../data/raw/tamil_processed/images', '../data/raw/tamil_processed/labels.txt')
    ]
    
    train_multilingual_model(
        language_configs=configs,
        num_epochs=50,
        batch_size=32
    )
