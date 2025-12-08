"""
Audio Analysis Models for Proctoring

Models for detecting:
- Multiple speakers/voices
- AI-generated speech
- Whispered speech
- Background conversations
- Suspicious audio patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioConfig:
    """Audio model configuration"""
    sample_rate: int = 16000
    n_mfcc: int = 40
    n_fft: int = 512
    hop_length: int = 160  # 10ms at 16kHz
    win_length: int = 400  # 25ms at 16kHz
    n_mels: int = 80
    
    # Model params
    hidden_size: int = 256
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    num_classes: int = 6  # normal, multiple_voices, whisper, ai_speech, background, suspicious


class AudioFeatureExtractor:
    """Extract audio features for model input"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self._init_transforms()
    
    def _init_transforms(self):
        """Initialize feature transforms"""
        try:
            import torchaudio
            import torchaudio.transforms as T
            
            self.mel_transform = T.MelSpectrogram(
                sample_rate=self.config.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
                n_mels=self.config.n_mels
            )
            
            self.mfcc_transform = T.MFCC(
                sample_rate=self.config.sample_rate,
                n_mfcc=self.config.n_mfcc,
                melkwargs={
                    'n_fft': self.config.n_fft,
                    'hop_length': self.config.hop_length,
                    'n_mels': self.config.n_mels
                }
            )
            
            self.use_torchaudio = True
        except ImportError:
            self.use_torchaudio = False
    
    def extract_features(
        self,
        waveform: torch.Tensor,
        feature_type: str = "mel"
    ) -> torch.Tensor:
        """
        Extract features from waveform.
        
        Args:
            waveform: Audio tensor [batch, samples] or [samples]
            feature_type: "mel", "mfcc", or "both"
            
        Returns:
            Features tensor [batch, channels, time]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        if self.use_torchaudio:
            if feature_type == "mel":
                features = self.mel_transform(waveform)
                features = torch.log(features + 1e-9)
            elif feature_type == "mfcc":
                features = self.mfcc_transform(waveform)
            else:  # both
                mel = torch.log(self.mel_transform(waveform) + 1e-9)
                mfcc = self.mfcc_transform(waveform)
                features = torch.cat([mel, mfcc], dim=1)
        else:
            # Fallback: simple spectrogram
            features = self._simple_spectrogram(waveform)
        
        return features
    
    def _simple_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Simple STFT-based spectrogram"""
        # Apply STFT
        spec = torch.stft(
            waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            return_complex=True
        )
        # Get magnitude
        mag = torch.abs(spec)
        # Log scale
        features = torch.log(mag + 1e-9)
        return features
    
    def extract_prosodic_features(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract prosodic features for voice analysis.
        
        Returns:
            Dict with pitch, energy, speaking_rate, etc.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        features = {}
        
        # Energy (RMS)
        frame_length = self.config.win_length
        hop = self.config.hop_length
        
        # Pad and reshape for framing
        num_frames = (waveform.shape[-1] - frame_length) // hop + 1
        
        energies = []
        for i in range(num_frames):
            start = i * hop
            frame = waveform[..., start:start + frame_length]
            energy = torch.sqrt(torch.mean(frame ** 2, dim=-1))
            energies.append(energy)
        
        features['energy'] = torch.stack(energies, dim=-1)
        
        # Zero crossing rate (indicator of voiced/unvoiced)
        zcr = []
        for i in range(num_frames):
            start = i * hop
            frame = waveform[..., start:start + frame_length]
            signs = torch.sign(frame)
            zcr_frame = torch.sum(torch.abs(signs[..., 1:] - signs[..., :-1]), dim=-1) / (2 * frame_length)
            zcr.append(zcr_frame)
        
        features['zcr'] = torch.stack(zcr, dim=-1)
        
        return features


class ConvBlock(nn.Module):
    """Convolutional block with batch norm and residual"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.residual = nn.Conv1d(in_channels, out_channels, 1, stride) if in_channels != out_channels or stride != 1 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        
        x = F.relu(x + residual)
        return x


class AudioEncoder(nn.Module):
    """
    CNN + Transformer encoder for audio features.
    """
    
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        
        # CNN layers for local feature extraction
        self.conv_layers = nn.Sequential(
            ConvBlock(config.n_mels, 64, kernel_size=3, stride=1),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            ConvBlock(128, 256, kernel_size=3, stride=2),
            ConvBlock(256, config.hidden_size, kernel_size=3, stride=2),
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, config.hidden_size) * 0.02)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        self.norm = nn.LayerNorm(config.hidden_size)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input features [batch, channels, time]
            mask: Optional attention mask
            
        Returns:
            Encoded features [batch, time', hidden_size]
        """
        # CNN encoding
        x = self.conv_layers(x)  # [batch, hidden, time']
        
        # Transpose for transformer
        x = x.transpose(1, 2)  # [batch, time', hidden]
        
        # Add positional encoding
        seq_len = x.shape[1]
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)
        
        return x


class VoiceClassifier(nn.Module):
    """
    Classify audio segments for proctoring.
    
    Classes:
    - normal_speech: Single speaker, normal interview
    - multiple_voices: More than one person speaking
    - whisper: Whispered/quiet speech
    - ai_speech: AI-generated speech patterns
    - background_speech: Background conversation
    - suspicious: Other suspicious patterns
    """
    
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        
        self.encoder = AudioEncoder(config)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_classes)
        )
        
        # Segment-level aggregation
        self.attention_pool = nn.Sequential(
            nn.Linear(config.hidden_size, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Audio features [batch, channels, time]
            
        Returns:
            Dict with logits, probabilities, attention weights
        """
        # Encode
        encoded = self.encoder(x)  # [batch, time, hidden]
        
        # Attention pooling
        attn_weights = self.attention_pool(encoded)  # [batch, time, 1]
        pooled = torch.sum(encoded * attn_weights, dim=1)  # [batch, hidden]
        
        # Classify
        logits = self.classifier(pooled)
        probs = F.softmax(logits, dim=-1)
        
        result = {
            'logits': logits,
            'probs': probs,
            'prediction': torch.argmax(probs, dim=-1)
        }
        
        if return_attention:
            result['attention'] = attn_weights.squeeze(-1)
            result['features'] = encoded
        
        return result


class SpeakerDiarization(nn.Module):
    """
    Detect and count speakers in audio.
    Uses speaker embeddings and clustering.
    """
    
    def __init__(self, config: AudioConfig, embedding_dim: int = 192):
        super().__init__()
        self.config = config
        self.embedding_dim = embedding_dim
        
        self.encoder = AudioEncoder(config)
        
        # Speaker embedding projection
        self.speaker_proj = nn.Sequential(
            nn.Linear(config.hidden_size, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Frame-level speaker prediction
        self.frame_classifier = nn.Linear(embedding_dim, 8)  # max 8 speakers
    
    def extract_embeddings(
        self,
        x: torch.Tensor,
        segment_length: int = 100
    ) -> torch.Tensor:
        """
        Extract speaker embeddings for each segment.
        
        Args:
            x: Audio features [batch, channels, time]
            segment_length: Frames per segment
            
        Returns:
            Embeddings [batch, num_segments, embedding_dim]
        """
        encoded = self.encoder(x)  # [batch, time, hidden]
        
        batch_size, time_len, hidden = encoded.shape
        num_segments = time_len // segment_length
        
        if num_segments == 0:
            # Use full sequence
            embeddings = self.speaker_proj(encoded.mean(dim=1, keepdim=True))
        else:
            # Split into segments
            segments = encoded[:, :num_segments * segment_length].reshape(
                batch_size, num_segments, segment_length, hidden
            )
            segment_means = segments.mean(dim=2)  # [batch, segments, hidden]
            embeddings = self.speaker_proj(segment_means)
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def count_speakers(
        self,
        x: torch.Tensor,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Estimate number of speakers in audio.
        
        Returns:
            Dict with speaker_count, embeddings, similarity_matrix
        """
        embeddings = self.extract_embeddings(x)  # [batch, segments, embed]
        
        batch_size = embeddings.shape[0]
        results = []
        
        for b in range(batch_size):
            emb = embeddings[b]  # [segments, embed]
            
            # Compute similarity matrix
            sim_matrix = torch.mm(emb, emb.t())  # [segments, segments]
            
            # Simple clustering: count groups above threshold
            # This is a simplified version - production would use spectral clustering
            visited = set()
            clusters = []
            
            for i in range(emb.shape[0]):
                if i in visited:
                    continue
                
                cluster = {i}
                for j in range(i + 1, emb.shape[0]):
                    if sim_matrix[i, j] > threshold:
                        cluster.add(j)
                        visited.add(j)
                
                clusters.append(cluster)
                visited.add(i)
            
            results.append({
                'speaker_count': len(clusters),
                'clusters': clusters,
                'similarity_matrix': sim_matrix
            })
        
        return results


class AIVoiceDetector(nn.Module):
    """
    Detect AI-generated speech.
    
    Looks for:
    - Unnatural prosody patterns
    - Perfect timing/rhythm
    - Lack of micro-variations
    - Spectral artifacts from synthesis
    """
    
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        
        self.encoder = AudioEncoder(config)
        
        # Artifact detection branch (high-frequency patterns)
        self.artifact_conv = nn.Sequential(
            nn.Conv1d(config.hidden_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Prosody analysis branch
        self.prosody_lstm = nn.LSTM(
            config.hidden_size,
            config.hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size + 64, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 2)  # real vs AI
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect if speech is AI-generated.
        
        Args:
            x: Audio features [batch, channels, time]
            
        Returns:
            Dict with is_ai probability and confidence
        """
        encoded = self.encoder(x)  # [batch, time, hidden]
        
        # Artifact features
        artifact_feat = self.artifact_conv(encoded.transpose(1, 2))  # [batch, 64, 1]
        artifact_feat = artifact_feat.squeeze(-1)  # [batch, 64]
        
        # Prosody features
        prosody_out, _ = self.prosody_lstm(encoded)
        prosody_feat = prosody_out.mean(dim=1)  # [batch, hidden]
        
        # Combine
        combined = torch.cat([prosody_feat, artifact_feat], dim=-1)
        logits = self.classifier(combined)
        
        probs = F.softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'is_ai_prob': probs[:, 1],
            'is_real_prob': probs[:, 0],
            'prediction': torch.argmax(probs, dim=-1)
        }


class AudioProctoringModel(nn.Module):
    """
    Combined audio proctoring model.
    
    Integrates:
    - Voice classification
    - Speaker diarization
    - AI voice detection
    """
    
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        
        # Shared encoder
        self.encoder = AudioEncoder(config)
        
        # Task heads
        self.voice_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_classes)
        )
        
        self.ai_detector = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        
        self.speaker_counter = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # 1-5 speakers
        )
        
        # Pooling
        self.attention_pool = nn.Linear(config.hidden_size, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        tasks: List[str] = ['classify', 'ai_detect', 'speaker_count']
    ) -> Dict[str, torch.Tensor]:
        """
        Run selected analysis tasks.
        """
        # Shared encoding
        encoded = self.encoder(x)  # [batch, time, hidden]
        
        # Attention pooling
        attn = F.softmax(self.attention_pool(encoded), dim=1)
        pooled = (encoded * attn).sum(dim=1)
        
        results = {}
        
        if 'classify' in tasks:
            class_logits = self.voice_classifier(pooled)
            results['class_logits'] = class_logits
            results['class_probs'] = F.softmax(class_logits, dim=-1)
            results['class_pred'] = torch.argmax(class_logits, dim=-1)
        
        if 'ai_detect' in tasks:
            ai_logits = self.ai_detector(pooled)
            results['ai_logits'] = ai_logits
            results['is_ai'] = F.softmax(ai_logits, dim=-1)[:, 1]
        
        if 'speaker_count' in tasks:
            speaker_logits = self.speaker_counter(pooled)
            results['speaker_logits'] = speaker_logits
            results['speaker_count'] = torch.argmax(speaker_logits, dim=-1) + 1
        
        return results


class AudioTrainer:
    """Trainer for audio models"""
    
    def __init__(
        self,
        model: nn.Module,
        config: AudioConfig,
        device: str = "auto"
    ):
        self.model = model
        self.config = config
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        self.feature_extractor = AudioFeatureExtractor(config)
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 50,
        lr: float = 1e-4,
        save_dir: str = "models/audio"
    ) -> Dict[str, List[float]]:
        """Train the audio model"""
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                waveforms, labels = batch
                waveforms = waveforms.to(self.device)
                labels = labels.to(self.device)
                
                # Extract features
                features = self.feature_extractor.extract_features(waveforms)
                
                # Forward
                optimizer.zero_grad()
                outputs = self.model(features)
                
                if isinstance(outputs, dict):
                    logits = outputs.get('logits') or outputs.get('class_logits')
                else:
                    logits = outputs
                
                loss = criterion(logits, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader:
                val_loss, val_acc = self._validate(val_loader, criterion)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(
                        self.model.state_dict(),
                        save_path / "best_model.pt"
                    )
                
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f} - "
                      f"Val Loss: {val_loss:.4f} - "
                      f"Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
            
            scheduler.step()
        
        # Save final model
        torch.save(self.model.state_dict(), save_path / "final_model.pt")
        
        return history
    
    def _validate(
        self,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Run validation"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                waveforms, labels = batch
                waveforms = waveforms.to(self.device)
                labels = labels.to(self.device)
                
                features = self.feature_extractor.extract_features(waveforms)
                outputs = self.model(features)
                
                if isinstance(outputs, dict):
                    logits = outputs.get('logits') or outputs.get('class_logits')
                else:
                    logits = outputs
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.shape[0]
        
        val_loss /= len(val_loader)
        accuracy = correct / total if total > 0 else 0
        
        return val_loss, accuracy
    
    def evaluate(
        self,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate model on test set"""
        from sklearn.metrics import classification_report, confusion_matrix
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                waveforms, labels = batch
                waveforms = waveforms.to(self.device)
                
                features = self.feature_extractor.extract_features(waveforms)
                outputs = self.model(features)
                
                if isinstance(outputs, dict):
                    preds = outputs.get('prediction') or outputs.get('class_pred')
                else:
                    preds = torch.argmax(outputs, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = (all_preds == all_labels).mean()
        
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
        
        return {
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels
        }
