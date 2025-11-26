import torch
from htr_model import HybridHTRModel
import json

class HTRModelLoader:
    def __init__(self, checkpoint_path, device='cpu'):
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.char_to_idx = checkpoint['char_to_idx']
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        
        # Extract num_classes from checkpoint model state
        # lstm.fc.weight has shape [num_classes, hidden*2]
        # lstm.fc.bias has shape [num_classes]
        num_classes = checkpoint['model_state_dict']['lstm.fc.bias'].shape[0]
        
        self.model = HybridHTRModel(num_classes=num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def decode_prediction(self, output):
        output = output.permute(1, 0, 2)
        output = torch.argmax(output, dim=2)
        output = output.squeeze(0).cpu().numpy()
        
        decoded = []
        prev_char = None
        for idx in output:
            if idx != 0 and idx != prev_char and idx in self.idx_to_char:
                decoded.append(self.idx_to_char[idx])
            prev_char = idx
        
        return ''.join(decoded)
    
    def predict(self, image_tensor):
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            output = self.model(image_tensor)
            prediction = self.decode_prediction(output)
        return prediction
