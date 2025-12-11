from __future__ import annotations
"""
Audio Predictor

Uses pre-trained models for robust audio proctoring:
- Silero VAD for voice activity detection
- Pyannote (optional) for speaker diarization  
- Resemblyzer for speaker embeddings
- Spectral analysis for AI voice detection

No training required - uses pre-trained weights.
"""

from typing import Any, Optional, Dict, List
from datetime import datetime
from collections import deque
import numpy as np

from proctor.engine.predictor import BasePredictor
from proctor.engine.results import AudioResults
from proctor.cfg import AudioConfig

# Lazy imports
torch = None
torchaudio = None


def _import_torch():
    """Lazy import torch."""
    global torch, torchaudio
    if torch is None:
        import torch as _torch
        torch = _torch
    if torchaudio is None:
        try:
            import torchaudio as _torchaudio
            torchaudio = _torchaudio
        except ImportError:
            torchaudio = None
    return torch, torchaudio


class AudioPredictor(BasePredictor):
    """
    Audio proctoring with pre-trained models.
    
    Detection capabilities:
    - Voice Activity Detection (Silero VAD)
    - Multiple Speaker Detection (embedding-based)
    - Whispering Detection (spectral analysis)
    - AI/Synthetic Voice Detection (spectral artifacts)
    - Background Conversation Detection
    - Suspicious Sounds (notifications, typing)
    """
    
    def __init__(self, cfg: AudioConfig):
        # Initialize model placeholders BEFORE super().__init__()
        # because super().__init__() calls setup_model() which sets these
        self.vad_model = None
        self.speaker_encoder = None
        
        # Buffers
        self.audio_buffer = deque(maxlen=cfg.sample_rate * 5)  # 5 second buffer
        self.embedding_buffer: List[np.ndarray] = []
        self.embedding_timestamps: List[datetime] = []
        
        # Speaker tracking
        self.primary_speaker_embedding: Optional[np.ndarray] = None
        self.speaker_embeddings: Dict[str, np.ndarray] = {}
        self.speaker_change_count = 0
        self.detected_speaker_count = 1
        
        # Statistics
        self.total_voice_duration_ms = 0
        self.total_silence_duration_ms = 0
        self.chunk_duration_ms = cfg.chunk_duration_ms
        
        # Calibration
        self.baseline_noise_level: Optional[float] = None
        self.baseline_spectral_centroid: Optional[float] = None
        self.calibrating = True
        self.calibration_samples: List[Dict] = []
        
        # Alert cooldown
        self.last_alerts: Dict[str, datetime] = {}
        
        # This calls setup_model() which populates self.vad_model, speaker_encoder
        super().__init__(cfg)
        
    def setup_model(self):
        """Load pre-trained models."""
        _import_torch()
        
        # Load Silero VAD
        self._load_silero_vad()
        
        # Load speaker encoder (for embeddings)
        self._load_speaker_encoder()
        
        print("✅ Audio Predictor initialized")
    
    def _load_silero_vad(self):
        """Load Silero VAD model."""
        try:
            # Use pip-installed silero-vad package
            from silero_vad import load_silero_vad
            self.vad_model = load_silero_vad()
            print("  ✓ Silero VAD loaded")
        except ImportError:
            # Fallback to torch.hub if package not installed
            try:
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    trust_repo=True
                )
                self.vad_model = model
                print("  ✓ Silero VAD loaded (torch.hub)")
            except Exception as e:
                print(f"  ⚠ Silero VAD failed: {e}")
                self.vad_model = None
        except Exception as e:
            print(f"  ⚠ Silero VAD failed: {e}")
            self.vad_model = None
    
    def _load_speaker_encoder(self):
        """Load speaker encoder for embeddings."""
        try:
            # Try to use resemblyzer for speaker embeddings
            from resemblyzer import VoiceEncoder
            self.speaker_encoder = VoiceEncoder()
            print("  ✓ Resemblyzer speaker encoder loaded")
        except ImportError:
            print("  ⚠ Resemblyzer not available, using MFCC-based embeddings")
            self.speaker_encoder = None
    
    def preprocess(self, source: Any) -> np.ndarray:
        """Preprocess audio to float32 [-1, 1]."""
        if isinstance(source, np.ndarray):
            audio = source.copy()
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
        
        # Convert to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Normalize
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val
        
        return audio
    
    def inference(self, audio: np.ndarray) -> Dict:
        """Run comprehensive audio analysis."""
        results = {}
        
        # Basic audio metrics
        results["audio_level_db"] = self._calculate_db(audio)
        results["noise_level"] = self._estimate_noise_level(audio)
        results["is_silent"] = results["audio_level_db"] < self.cfg.silence_threshold_db
        
        # Voice Activity Detection
        vad_result = self._detect_voice_activity(audio)
        results.update(vad_result)
        
        # Add to buffer for temporal analysis
        self.audio_buffer.extend(audio)
        
        # If voice detected, run additional analysis
        if results.get("voice_detected", False):
            self.total_voice_duration_ms += self.chunk_duration_ms
            
            # Whispering detection
            results["whispering"] = self._detect_whispering(
                audio, results["audio_level_db"]
            )
            
            # Speaker analysis (needs enough buffer)
            if len(self.audio_buffer) >= self.cfg.sample_rate:
                buffer_audio = np.array(self.audio_buffer)
                speaker_result = self._analyze_speakers(buffer_audio)
                results.update(speaker_result)
            else:
                results["speaker_count"] = 1
                results["multiple_speakers"] = False
                results["speaker_change_detected"] = False
            
            # AI voice detection
            results["ai_voice_probability"] = self._detect_ai_voice(audio)
            results["ai_voice_detected"] = results["ai_voice_probability"] > 0.7
            
            # Background voice detection
            results["background_voice"] = self._detect_background_voice(audio)
            
        else:
            self.total_silence_duration_ms += self.chunk_duration_ms
            results["whispering"] = False
            results["speaker_count"] = 0
            results["multiple_speakers"] = False
            results["speaker_change_detected"] = False
            results["ai_voice_probability"] = 0.0
            results["ai_voice_detected"] = False
            results["background_voice"] = False
        
        # Suspicious sounds detection
        results["suspicious_sounds"] = self._detect_suspicious_sounds(audio)
        
        return results
    
    def postprocess(self, preds: Dict, source: Any) -> AudioResults:
        """Convert predictions to AudioResults."""
        # Calibration
        if self.calibrating:
            self._update_calibration(preds)
        
        # Check for alerts
        should_alert, alert_type, severity = self._check_alerts(preds)
        
        return AudioResults(
            timestamp=datetime.utcnow(),
            source=source,
            voice_detected=preds.get("voice_detected", False),
            voice_confidence=preds.get("voice_confidence", 0.0),
            speaker_count=preds.get("speaker_count", 1),
            multiple_speakers=preds.get("multiple_speakers", False),
            speaker_change_detected=preds.get("speaker_change_detected", False),
            whispering_detected=preds.get("whispering", False),
            background_voice=preds.get("background_voice", False),
            suspicious_sounds=preds.get("suspicious_sounds", []),
            audio_level_db=preds.get("audio_level_db", -60.0),
            is_silent=preds.get("is_silent", True),
            noise_level=preds.get("noise_level", 0.0),
            should_alert=should_alert,
            alert_type=alert_type,
            alert_severity=severity,
        )
    
    # =========================================================================
    # Voice Activity Detection
    # =========================================================================
    
    def _detect_voice_activity(self, audio: np.ndarray) -> Dict:
        """Detect voice using Silero VAD."""
        if self.vad_model is None:
            # Fallback to energy-based
            db = self._calculate_db(audio)
            detected = db > self.cfg.silence_threshold_db
            confidence = min(1.0, max(0.0, (db + 60) / 40))
            return {"voice_detected": detected, "voice_confidence": confidence}
        
        try:
            audio_tensor = torch.from_numpy(audio).float()
            
            # Silero VAD expects specific sample rate
            if len(audio_tensor) < 512:
                # Pad short audio
                audio_tensor = torch.nn.functional.pad(
                    audio_tensor, (0, 512 - len(audio_tensor))
                )
            
            # Get speech probability
            speech_prob = self.vad_model(
                audio_tensor, 
                self.cfg.sample_rate
            ).item()
            
            return {
                "voice_detected": speech_prob > self.cfg.vad_threshold,
                "voice_confidence": float(speech_prob),
            }
        except Exception as e:
            # Fallback
            db = self._calculate_db(audio)
            return {
                "voice_detected": db > self.cfg.silence_threshold_db,
                "voice_confidence": 0.5,
            }
    
    # =========================================================================
    # Speaker Analysis
    # =========================================================================
    
    def _analyze_speakers(self, audio: np.ndarray) -> Dict:
        """Analyze speakers using embeddings."""
        result = {
            "speaker_count": 1,
            "multiple_speakers": False,
            "speaker_change_detected": False,
        }
        
        # Extract speaker embedding
        embedding = self._extract_speaker_embedding(audio)
        if embedding is None:
            return result
        
        # Store embedding with timestamp
        self.embedding_buffer.append(embedding)
        self.embedding_timestamps.append(datetime.utcnow())
        
        # Keep only recent embeddings (last 30 seconds)
        self._cleanup_old_embeddings(max_age_seconds=30)
        
        # Set primary speaker if not set
        if self.primary_speaker_embedding is None:
            self.primary_speaker_embedding = embedding
            return result
        
        # Compare to primary speaker
        similarity = self._cosine_similarity(embedding, self.primary_speaker_embedding)
        
        # Speaker change detection
        if similarity < 0.75:  # Different speaker threshold
            result["speaker_change_detected"] = True
            self.speaker_change_count += 1
        
        # Multiple speaker detection (analyze recent embeddings)
        if len(self.embedding_buffer) >= 3:
            unique_speakers = self._count_unique_speakers()
            result["speaker_count"] = unique_speakers
            result["multiple_speakers"] = unique_speakers > 1
        
        # Update primary speaker (moving average)
        if similarity > 0.85:
            self.primary_speaker_embedding = (
                0.9 * self.primary_speaker_embedding + 0.1 * embedding
            )
        
        return result
    
    def _extract_speaker_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract speaker embedding from audio."""
        if self.speaker_encoder is not None:
            try:
                # Use resemblyzer
                embedding = self.speaker_encoder.embed_utterance(audio)
                return embedding
            except Exception:
                pass
        
        # Fallback: MFCC-based embedding
        return self._extract_mfcc_embedding(audio)
    
    def _extract_mfcc_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC-based speaker embedding."""
        n_fft = 512
        hop_length = 256
        n_mels = 40
        n_mfcc = 20
        
        # Compute spectrogram
        spectrogram = []
        window = np.hanning(n_fft)
        
        for i in range(0, len(audio) - n_fft, hop_length):
            frame = audio[i:i+n_fft] * window
            fft = np.abs(np.fft.rfft(frame))
            spectrogram.append(fft)
        
        if not spectrogram:
            return np.zeros(n_mfcc * 2)
        
        spectrogram = np.array(spectrogram)
        
        # Mel filterbank
        mel_matrix = self._create_mel_filterbank(
            n_fft // 2 + 1, n_mels, self.cfg.sample_rate
        )
        mel_spec = np.dot(spectrogram, mel_matrix.T)
        log_mel = np.log(mel_spec + 1e-10)
        
        # DCT for MFCC
        mfcc = np.zeros((log_mel.shape[0], n_mfcc))
        for i in range(n_mfcc):
            for j in range(n_mels):
                mfcc[:, i] += log_mel[:, j] * np.cos(
                    np.pi * i * (j + 0.5) / n_mels
                )
        
        # Statistics as embedding
        mean_mfcc = np.mean(mfcc, axis=0)
        std_mfcc = np.std(mfcc, axis=0)
        
        embedding = np.concatenate([mean_mfcc, std_mfcc])
        return embedding / (np.linalg.norm(embedding) + 1e-10)
    
    def _count_unique_speakers(self) -> int:
        """Count unique speakers from recent embeddings."""
        if len(self.embedding_buffer) < 2:
            return 1
        
        embeddings = np.array(self.embedding_buffer[-10:])  # Last 10
        
        # Simple clustering: count groups with similarity < threshold
        n = len(embeddings)
        visited = set()
        clusters = 0
        
        for i in range(n):
            if i in visited:
                continue
            clusters += 1
            visited.add(i)
            
            for j in range(i + 1, n):
                if j in visited:
                    continue
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                if sim > 0.75:
                    visited.add(j)
        
        return min(clusters, 5)  # Cap at 5
    
    def _cleanup_old_embeddings(self, max_age_seconds: int = 30):
        """Remove old embeddings from buffer."""
        now = datetime.utcnow()
        while (self.embedding_timestamps and 
               (now - self.embedding_timestamps[0]).total_seconds() > max_age_seconds):
            self.embedding_buffer.pop(0)
            self.embedding_timestamps.pop(0)
    
    # =========================================================================
    # Whispering Detection
    # =========================================================================
    
    def _detect_whispering(self, audio: np.ndarray, db_level: float) -> bool:
        """Detect whispering using spectral analysis."""
        # Whispering characteristics:
        # 1. Low amplitude but voice detected
        # 2. Reduced low-frequency energy
        # 3. More high-frequency noise (aperiodic)
        
        if db_level > self.cfg.normal_speech_db:
            return False  # Too loud for whisper
        
        if db_level < self.cfg.silence_threshold_db:
            return False  # Too quiet
        
        # Spectral analysis
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/self.cfg.sample_rate)
        
        # Energy in frequency bands
        low_band = (freqs >= 100) & (freqs <= 500)
        mid_band = (freqs >= 500) & (freqs <= 2000)
        high_band = (freqs >= 2000) & (freqs <= 6000)
        
        low_energy = np.sum(fft[low_band] ** 2)
        mid_energy = np.sum(fft[mid_band] ** 2)
        high_energy = np.sum(fft[high_band] ** 2)
        
        total_energy = low_energy + mid_energy + high_energy + 1e-10
        
        # Whisper has less low-frequency energy and more high-frequency
        low_ratio = low_energy / total_energy
        high_ratio = high_energy / total_energy
        
        # Whisper detection: low fundamentals, high noise
        is_whisper = (
            low_ratio < 0.2 and  # Reduced low frequencies
            high_ratio > 0.3 and  # Elevated high frequencies
            db_level < self.cfg.whisper_threshold_db
        )
        
        return is_whisper
    
    # =========================================================================
    # AI Voice Detection
    # =========================================================================
    
    def _detect_ai_voice(self, audio: np.ndarray) -> float:
        """
        Detect AI-generated voice using spectral analysis.
        
        AI voices often have:
        - Too perfect pitch consistency
        - Unnatural spectral smoothness
        - Missing micro-variations
        - Specific frequency artifacts from vocoders
        """
        if len(audio) < 1024:
            return 0.0
        
        indicators = []
        
        # 1. Spectral flatness (AI often has smoother spectra)
        fft = np.abs(np.fft.rfft(audio))
        geometric_mean = np.exp(np.mean(np.log(fft + 1e-10)))
        arithmetic_mean = np.mean(fft)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        
        # Natural speech has lower flatness than synthetic
        if spectral_flatness > 0.3:
            indicators.append(0.6)  # Suspiciously flat
        else:
            indicators.append(0.0)
        
        # 2. Pitch variance (AI has unnaturally consistent pitch)
        pitch_variance = self._estimate_pitch_variance(audio)
        if pitch_variance < 5:  # Very consistent pitch
            indicators.append(0.7)
        elif pitch_variance < 15:
            indicators.append(0.3)
        else:
            indicators.append(0.0)
        
        # 3. Micro-pause analysis (AI has unnatural timing)
        energy_variance = self._compute_energy_variance(audio)
        if energy_variance < 0.01:  # Too consistent
            indicators.append(0.5)
        else:
            indicators.append(0.0)
        
        # 4. High-frequency artifacts (vocoder signatures)
        freqs = np.fft.rfftfreq(len(audio), 1/self.cfg.sample_rate)
        ultra_high = freqs > 7000
        if np.any(ultra_high):
            high_energy_ratio = np.sum(fft[ultra_high]**2) / (np.sum(fft**2) + 1e-10)
            if high_energy_ratio < 0.001:  # Suspiciously clean cutoff
                indicators.append(0.4)
            else:
                indicators.append(0.0)
        
        # Combine indicators
        return min(1.0, np.mean(indicators) * 1.5) if indicators else 0.0
    
    def _estimate_pitch_variance(self, audio: np.ndarray) -> float:
        """Estimate pitch variance using autocorrelation."""
        # Simple autocorrelation-based pitch estimation
        n = len(audio)
        if n < 512:
            return 50.0  # Default to normal
        
        # Compute autocorrelation for different lags
        min_period = int(self.cfg.sample_rate / 500)  # Max 500 Hz
        max_period = int(self.cfg.sample_rate / 50)   # Min 50 Hz
        
        pitches = []
        hop = 256
        
        for start in range(0, n - max_period - 1, hop):
            frame = audio[start:start + max_period]
            if len(frame) < max_period:
                continue
                
            correlations = []
            for lag in range(min_period, min(max_period, len(frame))):
                if lag >= len(frame):
                    break
                corr = np.sum(frame[:len(frame)-lag] * frame[lag:])
                correlations.append((lag, corr))
            
            if correlations:
                best_lag = max(correlations, key=lambda x: x[1])[0]
                pitch = self.cfg.sample_rate / best_lag
                if 50 < pitch < 500:
                    pitches.append(pitch)
        
        if len(pitches) > 2:
            return float(np.std(pitches))
        return 50.0
    
    def _compute_energy_variance(self, audio: np.ndarray) -> float:
        """Compute frame-level energy variance."""
        frame_size = 256
        energies = []
        
        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i+frame_size]
            energy = np.sqrt(np.mean(frame ** 2))
            energies.append(energy)
        
        if energies:
            return float(np.var(energies))
        return 0.1
    
    # =========================================================================
    # Background Voice Detection
    # =========================================================================
    
    def _detect_background_voice(self, audio: np.ndarray) -> bool:
        """Detect background conversation."""
        # Look for overlapping speech patterns
        
        # 1. Check for speech-like energy in multiple frequency bands simultaneously
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/self.cfg.sample_rate)
        
        # Voice formant regions
        f1_band = (freqs >= 200) & (freqs <= 800)   # First formant
        f2_band = (freqs >= 800) & (freqs <= 2500)  # Second formant
        f3_band = (freqs >= 2500) & (freqs <= 4000) # Third formant
        
        f1_energy = np.sum(fft[f1_band] ** 2)
        f2_energy = np.sum(fft[f2_band] ** 2)
        f3_energy = np.sum(fft[f3_band] ** 2)
        
        total = f1_energy + f2_energy + f3_energy + 1e-10
        
        # Check for unusual distribution (overlapping voices)
        f1_ratio = f1_energy / total
        f2_ratio = f2_energy / total
        
        # Overlapping speech often has energy spread more evenly
        # and higher total energy
        db_level = self._calculate_db(audio)
        
        # Strong energy in all bands suggests multiple voices
        if f1_ratio > 0.2 and f2_ratio > 0.2 and db_level > -25:
            # Additional check: detect rapid energy fluctuations
            energy_fluct = self._compute_energy_variance(audio)
            if energy_fluct > 0.05:
                return True
        
        return False
    
    # =========================================================================
    # Suspicious Sounds Detection
    # =========================================================================
    
    def _detect_suspicious_sounds(self, audio: np.ndarray) -> List[str]:
        """Detect suspicious sounds like notifications or typing."""
        sounds = []
        
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/self.cfg.sample_rate)
        
        # 1. Notification sounds (tonal, 1-4kHz range)
        tone_band = (freqs >= 800) & (freqs <= 4000)
        if np.any(tone_band):
            tone_fft = fft[tone_band]
            tone_freqs = freqs[tone_band]
            
            # Look for narrow spectral peaks (tones)
            if len(tone_fft) > 10:
                peak_idx = np.argmax(tone_fft)
                peak_energy = tone_fft[peak_idx]
                mean_energy = np.mean(tone_fft)
                
                # Sharp peak indicates tone
                if peak_energy > mean_energy * 5:
                    sounds.append("notification_tone")
        
        # 2. Typing sounds (transients, broadband)
        envelope = np.abs(audio)
        diff = np.diff(envelope)
        transients = np.sum(np.abs(diff) > 0.1)
        
        # Many transients in short audio = typing
        transient_rate = transients / (len(audio) / self.cfg.sample_rate)
        if transient_rate > 20:  # More than 20 transients per second
            sounds.append("typing")
        
        # 3. Page turning (brief broadband burst)
        if transient_rate > 5 and transient_rate < 15:
            # Check for broadband energy
            low_e = np.sum(fft[freqs < 500] ** 2)
            high_e = np.sum(fft[freqs > 2000] ** 2)
            if high_e > low_e * 0.5:
                sounds.append("paper_rustle")
        
        return sounds
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def _calculate_db(self, audio: np.ndarray) -> float:
        """Calculate audio level in decibels."""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            return float(20 * np.log10(rms))
        return -100.0
    
    def _estimate_noise_level(self, audio: np.ndarray) -> float:
        """Estimate background noise level."""
        window_size = len(audio) // 10
        if window_size < 100:
            return float(np.sqrt(np.mean(audio ** 2)))
        
        rms_values = []
        for i in range(0, len(audio) - window_size, window_size):
            window = audio[i:i+window_size]
            rms_values.append(np.sqrt(np.mean(window ** 2)))
        
        if rms_values:
            return float(np.percentile(rms_values, 10))
        return 0.0
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def _create_mel_filterbank(
        self, n_fft: int, n_mels: int, sample_rate: int
    ) -> np.ndarray:
        """Create mel filterbank matrix."""
        low_freq = 0
        high_freq = sample_rate / 2
        
        low_mel = 2595 * np.log10(1 + low_freq / 700)
        high_mel = 2595 * np.log10(1 + high_freq / 700)
        
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((n_fft) * hz_points / sample_rate).astype(int)
        
        filterbank = np.zeros((n_mels, n_fft))
        
        for i in range(1, n_mels + 1):
            left = bin_points[i-1]
            center = bin_points[i]
            right = bin_points[i+1]
            
            for j in range(left, center):
                if center != left:
                    filterbank[i-1, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right != center:
                    filterbank[i-1, j] = (right - j) / (right - center)
        
        return filterbank
    
    def _update_calibration(self, preds: Dict):
        """Update calibration with new sample."""
        self.calibration_samples.append({
            "noise_level": preds.get("noise_level", 0),
            "audio_level_db": preds.get("audio_level_db", -60),
        })
        
        if len(self.calibration_samples) >= self.cfg.calibration_samples:
            noise_levels = [s["noise_level"] for s in self.calibration_samples]
            self.baseline_noise_level = float(np.median(noise_levels))
            self.calibrating = False
            print(f"✅ Audio baseline calibrated: noise_level={self.baseline_noise_level:.4f}")
    
    def _check_alerts(self, preds: Dict) -> tuple:
        """Check if any alerts should be triggered."""
        now = datetime.utcnow()
        cooldown = self.cfg.alert_cooldown_seconds
        
        def can_alert(alert_type: str) -> bool:
            if alert_type not in self.last_alerts:
                return True
            elapsed = (now - self.last_alerts[alert_type]).total_seconds()
            return elapsed >= cooldown
        
        # Priority order of alerts
        if preds.get("multiple_speakers") and can_alert("multiple_speakers"):
            self.last_alerts["multiple_speakers"] = now
            return True, "multiple_speakers", "high"
        
        if preds.get("background_voice") and can_alert("background_voice"):
            self.last_alerts["background_voice"] = now
            return True, "background_voice", "high"
        
        if preds.get("ai_voice_detected") and can_alert("ai_voice"):
            self.last_alerts["ai_voice"] = now
            return True, "ai_voice_detected", "high"
        
        if preds.get("whispering") and can_alert("whispering"):
            self.last_alerts["whispering"] = now
            return True, "whispering", "medium"
        
        suspicious = preds.get("suspicious_sounds", [])
        if "notification_tone" in suspicious and can_alert("notification"):
            self.last_alerts["notification"] = now
            return True, "phone_notification", "medium"
        
        if "typing" in suspicious and can_alert("typing"):
            self.last_alerts["typing"] = now
            return True, "typing_detected", "low"
        
        return False, None, "low"
    
    def calibrate(self, audio_samples: np.ndarray, sample_rate: int = 16000):
        """Manually calibrate baseline."""
        audio = self.preprocess(audio_samples)
        self.baseline_noise_level = self._estimate_noise_level(audio)
        self.calibrating = False
        print(f"✅ Audio baseline set: noise_level={self.baseline_noise_level:.4f}")
    
    def reset(self):
        """Reset predictor state."""
        self.audio_buffer.clear()
        self.embedding_buffer.clear()
        self.embedding_timestamps.clear()
        self.primary_speaker_embedding = None
        self.speaker_change_count = 0
        self.detected_speaker_count = 1
        self.total_voice_duration_ms = 0
        self.total_silence_duration_ms = 0
        self.baseline_noise_level = None
        self.calibrating = True
        self.calibration_samples.clear()
        self.last_alerts.clear()
    
    def get_statistics(self) -> Dict:
        """Get audio analysis statistics."""
        total_duration = self.total_voice_duration_ms + self.total_silence_duration_ms
        
        return {
            "total_voice_duration_ms": self.total_voice_duration_ms,
            "total_silence_duration_ms": self.total_silence_duration_ms,
            "voice_ratio": self.total_voice_duration_ms / total_duration if total_duration > 0 else 0,
            "speaker_changes": self.speaker_change_count,
            "detected_speakers": self.detected_speaker_count,
            "baseline_noise_level": self.baseline_noise_level,
        }
