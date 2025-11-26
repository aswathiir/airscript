import numpy as np
from PIL import Image
import torch
from torchvision import transforms

class StrokeProcessor:
    def __init__(self, image_width=256, image_height=64):
        self.image_width = image_width
        self.image_height = image_height
        self.transform = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def strokes_to_image(self, strokes):
        if not strokes or len(strokes) == 0:
            return None
        
        all_x = [point['x'] for stroke in strokes for point in stroke]
        all_y = [point['y'] for stroke in strokes for point in stroke]
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        width = max(max_x - min_x, 1)
        height = max(max_y - min_y, 1)
        
        scale = min(self.image_width / width, self.image_height / height) * 0.9
        
        canvas = np.ones((self.image_height, self.image_width), dtype=np.uint8) * 255
        
        for stroke in strokes:
            for i in range(len(stroke) - 1):
                x1 = int((stroke[i]['x'] - min_x) * scale)
                y1 = int((stroke[i]['y'] - min_y) * scale)
                x2 = int((stroke[i+1]['x'] - min_x) * scale)
                y2 = int((stroke[i+1]['y'] - min_y) * scale)
                
                self._draw_line(canvas, x1, y1, x2, y2)
        
        return Image.fromarray(canvas, mode='L')
    
    def _draw_line(self, canvas, x1, y1, x2, y2, thickness=2):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while True:
            for i in range(-thickness, thickness+1):
                for j in range(-thickness, thickness+1):
                    px, py = x1 + i, y1 + j
                    if 0 <= px < self.image_width and 0 <= py < self.image_height:
                        canvas[py, px] = 0
            
            if x1 == x2 and y1 == y2:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
    
    def prepare_input(self, image):
        tensor = self.transform(image)
        return tensor.unsqueeze(0)
