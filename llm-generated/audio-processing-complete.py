"""
10_audio_processing.py
Comprehensive Audio Processing Pipeline
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class AudioPreprocessor:
    """Comprehensive audio preprocessing pipeline"""
    
    def __init__(self, target_sr=22050, duration=3, n_mfcc=13):
        self.target_sr = target_sr
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.scaler = StandardScaler()
        
    def load_audio(self, file_path, resample=True):
        """Load and optionally resample audio file"""
        try:
            audio, sr = librosa.load(file_path, sr=self.target_sr if resample else None)
            return audio, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, self.target_sr
    
    def extract_features(self, audio, sr):
        """Extract comprehensive audio features"""
        features = {}
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features['chroma_mean'] = np.mean(chroma, axis=1)
        features['chroma_std'] = np.std(chroma, axis=1)
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        features['mel_spec_mean'] = np.mean(mel_spec)
        features['mel_spec_std'] = np.std(mel_spec)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        features['tempo'] = tempo
        
        return features
    
    def create_feature_vector(self, features_dict):
        """Convert features dictionary to numpy array"""
        feature_vector = []
        for key in sorted(features_dict.keys()):
            if isinstance(features_dict[key], np.ndarray):
                feature_vector.extend(features_dict[key])
            else:
                feature_vector.append(features_dict[key])
        return np.array(feature_vector)
    
    def preprocess_audio_files(self, file_paths, labels=None):
        """Preprocess multiple audio files"""
        features_list = []
        valid_files = []
        valid_labels = [] if labels is not None else None
        
        for i, file_path in enumerate(file_paths):
            audio, sr = self.load_audio(file_path)
            if audio is not None:
                # Ensure fixed length
                target_length = self.target_sr * self.duration
                if len(audio) > target_length:
                    audio = audio[:target_length]
                else:
                    audio = np.pad(audio, (0, max(0, target_length - len(audio))))
                
                features = self.extract_features(audio, sr)
                feature_vector = self.create_feature_vector(features)
                features_list.append(feature_vector)
                valid_files.append(file_path)
                
                if labels is not None:
                    valid_labels.append(labels[i])
        
        features_array = np.array(features_list)
        
        # Scale features
        if len(features_array) > 0:
            features_array = self.scaler.fit_transform(features_array)
        
        return features_array, valid_files, valid_labels

    def create_mel_spectrogram(self, audio, sr, n_mels=128, fmax=8000):
        """Create mel spectrogram for deep learning"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, fmax=fmax
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db

    def extract_mfcc_sequence(self, audio, sr, n_mfcc=13, hop_length=512):
        """Extract MFCC sequence for sequence models"""
        mfccs = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length
        )
        return mfccs.T  # Transpose to (time_steps, n_mfcc)

class AudioDataset(Dataset):
    """PyTorch Dataset for audio files"""
    
    def __init__(self, file_paths, labels=None, transform=None, target_sr=22050, duration=3, feature_type='waveform'):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.target_sr = target_sr
        self.duration = duration
        self.target_length = target_sr * duration
        self.feature_type = feature_type
        self.preprocessor = AudioPreprocessor(target_sr, duration)
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        if self.feature_type == 'waveform':
            # Load raw audio
            audio, sr = torchaudio.load(file_path)
            
            # Resample if necessary
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                audio = resampler(audio)
            
            # Convert to mono
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Ensure fixed length
            if audio.shape[1] > self.target_length:
                audio = audio[:, :self.target_length]
            else:
                padding = self.target_length - audio.shape[1]
                audio = torch.nn.functional.pad(audio, (0, padding))
            
            # Apply transforms
            if self.transform:
                audio = self.transform(audio)
                
            features = audio
            
        elif self.feature_type == 'melspectrogram':
            # Load and create mel spectrogram
            audio, sr = librosa.load(file_path, sr=self.target_sr)
            
            # Ensure fixed length
            target_samples = self.target_sr * self.duration
            if len(audio) > target_samples:
                audio = audio[:target_samples]
            else:
                audio = np.pad(audio, (0, max(0, target_samples - len(audio))))
            
            # Create mel spectrogram
            mel_spec = self.preprocessor.create_mel_spectrogram(audio, sr)
            features = torch.FloatTensor(mel_spec).unsqueeze(0)  # Add channel dimension
            
        elif self.feature_type == 'mfcc':
            # Load and create MFCCs
            audio, sr = librosa.load(file_path, sr=self.target_sr)
            
            # Ensure fixed length
            target_samples = self.target_sr * self.duration
            if len(audio) > target_samples:
                audio = audio[:target_samples]
            else:
                audio = np.pad(audio, (0, max(0, target_samples - len(audio))))
            
            # Create MFCCs
            mfccs = self.preprocessor.extract_mfcc_sequence(audio, sr)
            features = torch.FloatTensor(mfccs.T)  # Shape: (n_mfcc, time_steps)
            features = features.unsqueeze(0)  # Add channel dimension
        
        if self.labels is not None:
            return features, self.labels[idx]
        else:
            return features

class AudioCNN(nn.Module):
    """CNN for audio classification from spectrograms"""
    
    def __init__(self, num_classes, input_channels=1):
        super(AudioCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 512),  # Adjust based on input size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class AudioLSTM(nn.Module):
    """LSTM for audio sequence classification"""
    
    def __init__(self, num_classes, input_size=13, hidden_size=128, num_layers=2, dropout=0.3):
        super(AudioLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features) or (batch, features, seq_len)
        if x.dim() == 3 and x.size(1) != self.lstm.input_size:
            x = x.transpose(1, 2)  # Convert to (batch, seq_len, features)
        
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.classifier(last_hidden)
        
        return output

class AudioTransformer(nn.Module):
    """Transformer model for audio processing"""
    
    def __init__(self, num_classes, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super(AudioTransformer, self).__init__()
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(128, d_model, kernel_size=5, stride=2),
            nn.ReLU()
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, 1, seq_len)
        x = self.feature_extractor(x)  # (batch, d_model, reduced_seq_len)
        x = x.transpose(1, 2)  # (batch, reduced_seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classify
        x = self.classifier(x)
        
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class AudioAugmentation:
    """Audio data augmentation techniques"""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
    
    def time_shift(self, audio, shift_limit=0.4):
        """Time shifting augmentation"""
        shift = int(np.random.uniform(-shift_limit, shift_limit) * len(audio))
        if shift > 0:
            audio = np.pad(audio, (shift, 0), mode='constant')[:-shift]
        else:
            audio = np.pad(audio, (0, -shift), mode='constant')[-shift:]
        return audio
    
    def pitch_shift(self, audio, n_steps=2):
        """Pitch shifting augmentation"""
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
    
    def time_stretch(self, audio, rate=1.2):
        """Time stretching augmentation"""
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def add_noise(self, audio, noise_level=0.005):
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_level, audio.shape)
        return audio + noise
    
    def apply_random_gain(self, audio, min_gain=0.5, max_gain=1.5):
        """Random gain adjustment"""
        gain = np.random.uniform(min_gain, max_gain)
        return audio * gain

class SpeechRecognitionModel(nn.Module):
    """ASR model using CNN + RNN + CTC"""
    
    def __init__(self, num_chars, hidden_size=256, num_layers=3, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        
        # Feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # RNN layers
        self.rnn = nn.LSTM(
            input_size=64 * 20,  # Adjust based on conv output
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Output layer
        self.output = nn.Linear(hidden_size * 2, num_chars)
    
    def forward(self, x):
        # x shape: (batch, 1, features, time)
        x = self.conv_layers(x)
        
        # Reshape for RNN
        batch_size, channels, features, time_steps = x.size()
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, features)
        x = x.contiguous().view(batch_size, time_steps, -1)
        
        # RNN
        x, _ = self.rnn(x)
        
        # Output
        x = self.output(x)
        return x

class AudioTrainer:
    """Audio model trainer"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def train(self, train_loader, val_loader, epochs=50, lr=0.001):
        """Train audio model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
            
            # Validation
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            scheduler.step(val_loss)
            
            train_loss_avg = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            train_losses.append(train_loss_avg)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
    
    def evaluate(self, data_loader, criterion):
        """Evaluate model"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = test_loss / len(data_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy

class AudioPipeline:
    """Complete audio processing pipeline"""
    
    def __init__(self, feature_type='melspectrogram', target_sr=22050, duration=3):
        self.feature_type = feature_type
        self.target_sr = target_sr
        self.duration = duration
        self.preprocessor = AudioPreprocessor(target_sr, duration)
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess(self, file_paths, labels=None):
        """Load and preprocess audio files"""
        if self.feature_type in ['waveform', 'melspectrogram', 'mfcc']:
            # For deep learning models
            dataset = AudioDataset(
                file_paths, labels, 
                target_sr=self.target_sr,
                duration=self.duration,
                feature_type=self.feature_type
            )
            return dataset
        else:
            # For traditional ML
            features, valid_files, valid_labels = self.preprocessor.preprocess_audio_files(file_paths, labels)
            return features, valid_labels
    
    def create_model(self, num_classes, model_type='cnn'):
        """Create audio model"""
        if model_type == 'cnn':
            self.model = AudioCNN(num_classes)
        elif model_type == 'lstm':
            self.model = AudioLSTM(num_classes)
        elif model_type == 'transformer':
            self.model = AudioTransformer(num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return self.model
    
    def train(self, train_dataset, val_dataset, model_type='cnn', epochs=50):
        """Train the audio model"""
        if self.model is None:
            num_classes = len(np.unique([label for _, label in train_dataset]))
            self.create_model(num_classes, model_type)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = AudioTrainer(self.model, device)
        
        history = trainer.train(train_loader, val_loader, epochs=epochs)
        return history
    
    def predict(self, file_paths):
        """Predict on new audio files"""
        if self.model is None:
            raise ValueError("Model must be trained first!")
        
        dataset = AudioDataset(
            file_paths, 
            target_sr=self.target_sr,
            duration=self.duration,
            feature_type=self.feature_type
        )
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for data in dataloader:
                if torch.is_tensor(data):
                    data = data.to(next(self.model.parameters()).device)
                else:
                    data = data[0].to(next(self.model.parameters()).device)
                
                output = self.model(data)
                pred = output.argmax(dim=1)
                predictions.extend(pred.cpu().numpy())
        
        return predictions

# Example usage
if __name__ == "__main__":
    # Example usage
    pipeline = AudioPipeline(feature_type='melspectrogram')
    
    print("Audio Processing Pipeline Ready!")
    print("Available features:")
    print("- waveform: Raw audio waveforms")
    print("- melspectrogram: Mel spectrograms (recommended for CNN)")
    print("- mfcc: MFCC sequences (good for LSTM/Transformer)")
    
    print("\nUsage:")
    print("1. pipeline.load_and_preprocess(file_paths, labels)")
    print("2. pipeline.create_model(num_classes, 'cnn')")
    print("3. pipeline.train(train_dataset, val_dataset, epochs=50)")
    print("4. predictions = pipeline.predict(test_files)")
