import os

def verify_processed_dataset(lang_name, processed_dir):
    """Verify processed dataset structure"""
    
    labels_path = os.path.join(processed_dir, 'labels.txt')
    images_dir = os.path.join(processed_dir, 'images')
    
    if not os.path.exists(labels_path):
        print(f'{lang_name}: ✗ labels.txt missing')
        return False
    
    if not os.path.exists(images_dir):
        print(f'{lang_name}: ✗ images/ folder missing')
        return False
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    
    image_files = os.listdir(images_dir)
    
    print(f'{lang_name}:')
    print(f'  Total samples: {len(labels)}')
    print(f'  Image files: {len(image_files)}')
    
    if len(labels) != len(image_files):
        print(f'  ✗ Mismatch: {len(labels)} labels vs {len(image_files)} images')
        return False
    
    # Verify label format
    sample_labels = labels[:3]
    print(f'  Sample labels:')
    for label in sample_labels:
        parts = label.split('\t')
        print(f'    {parts[0]} → {parts[1] if len(parts) > 1 else "NO_LABEL"}')
    
    print(f'  ✓ Valid dataset\n')
    return True

if __name__ == '__main__':
    base_dir = '../data/raw'
    
    verify_processed_dataset('Hindi', os.path.join(base_dir, 'hindi_processed'))
    verify_processed_dataset('Malayalam', os.path.join(base_dir, 'malayalam_processed'))
    verify_processed_dataset('Tamil', os.path.join(base_dir, 'tamil_processed'))
