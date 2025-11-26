import torch
import torch.nn as nn

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1))
        )
        
    def forward(self, x):
        features = self.conv_layers(x)
        b, c, h, w = features.size()
        features = features.view(b, c * h, w)
        features = features.permute(0, 2, 1)
        return features

class BiLSTMSequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTMSequenceModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

class HybridHTRModel(nn.Module):
    def __init__(self, num_classes, lstm_hidden=256, lstm_layers=2):
        super(HybridHTRModel, self).__init__()
        self.cnn = CNNFeatureExtractor()
        # Determine the correct LSTM input size dynamically by running a
        # dummy tensor through the CNN feature extractor. This ensures the
        # LSTM input_size matches c*h (channels * feature_height) for the
        # configured input image size used in training (64x256 by default).
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 64, 256)
            feat = self.cnn(dummy)
            # feat shape: (batch, seq_len, feat_dim) where feat_dim == c*h
            input_size = feat.size(2)

        self.lstm = BiLSTMSequenceModel(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            num_classes=num_classes
        )
        
    def forward(self, x):
        cnn_features = self.cnn(x)
        output = self.lstm(cnn_features)
        return output
