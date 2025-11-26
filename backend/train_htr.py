import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from htr_model import HybridHTRModel
from data_loader import IndicHTRDataset, collate_fn
from tqdm import tqdm
import os

def train_model(
    data_dir,
    label_file,
    language_name='Dataset',
    num_epochs=50,
    batch_size=32,
    learning_rate=0.001,
    save_dir='../models/htr',
    checkpoint_resume=None
):
    os.makedirs(save_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    print(f'\n{"="*60}')
    print(f'Loading {language_name} dataset...')
    print(f'{"="*60}')
    
    dataset = IndicHTRDataset(data_dir, label_file, transform=transform)
    print(f'Dataset size: {len(dataset)}')
    print(f'Vocabulary size: {len(dataset.char_to_idx)}')
    print(f'Characters: {list(dataset.char_to_idx.keys())[:10]}...')
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    num_classes = len(dataset.char_to_idx)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'Training device: {device}')
    print(f'Output classes: {num_classes}')
    
    model = HybridHTRModel(num_classes=num_classes).to(device)
    
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    start_epoch = 0
    
    if checkpoint_resume and os.path.exists(checkpoint_resume):
        print(f'Resuming from checkpoint: {checkpoint_resume}')
        checkpoint = torch.load(checkpoint_resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    print(f'\n{"="*60}')
    print(f'Starting training for {num_epochs} epochs')
    print(f'{"="*60}\n')
    
    for epoch in range(start_epoch, num_epochs):
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
            checkpoint_path_2 = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'char_to_idx': dataset.char_to_idx
            }, checkpoint_path_2)
            print(f'Checkpoint saved: {checkpoint_path_2}')
    
    final_path = os.path.join(save_dir, 'checkpoint_finetuned.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': dataset.char_to_idx
    }, final_path)
    print(f'\nFinal model saved: {final_path}')

if __name__ == '__main__':
    # Train on Hindi first (has most reliable labels)
    train_model(
        data_dir='../data/raw/malayalam_processed/images',
        label_file='../data/raw/malayalam_processed/labels.txt',
        language_name='Malayalam',
        num_epochs=50,
        batch_size=32
    )
