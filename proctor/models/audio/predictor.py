"""
Audio Predictor

Handles audio analysis with VAD and speaker detection.
"""

from typing import Any, Optional
from datetime import datetime
from collections import deque
import numpy as np

from proctor.engine.predictor import BasePredictor
from proctor.engine.results import AudioResults
from proctor.cfg import AudioConfig

# Lazy imports
torch = None
torchaudio = None
silero_vad = None


def _import_torch():
    """Lazy import torch."""
    global torch, torchaudio
    if torch is None:
        import torch as _torch
        torch = _torch
    if torchaudio is None:
        import torchaudio as _torchaudio
        torchaudio = _torchaudio
    return torch, torchaudio


def _import_silero():
    """Lazy import Silero VAD."""
    global silero_vad
    if silero_vad is None:
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps
            silero_vad = {"load": load_silero_vad, "timestamps": get_speech_timestamps}
        except ImportError:
            silero_vad = None
    return silero_vad


class AudioPredictor(BasePredictor):
    """
    Audio analysis predictor.
    
    Performs:
    - Voice Activity Detection (VAD)
    - Speaker counting
    - Whispering detection
    - Background voice detection
    - Suspicious sound detection
    """
    
    def __init__(self, cfg: AudioConfig):
        """
        Initialize audio predictor.
        
        Args:
            cfg: Audio configuration
        """
        super().__init__(cfg)
        
        self.vad_model = None
        self.audio_buffer = deque(maxlen=32000)  # 2 seconds buffer
        
        # Speaker tracking
        self.primary_speaker_embedding = None
        self.speaker_change_count = 0
        
        # Statistics
        self.total_voice_duration = 0
        self.total_silence_duration = 0
        
        # Baseline calibration
        self.baseline_noise_level: Optional[float] = None
        self.calibrating = True
        self.calibration_samples: list[float] = []
    
    def setup_model(self):
        """Load VAD model."""
        silero = _import_silero()
        
        if silero:
            try:
                self.vad_model = silero["load"]()
                print("✅ Silero VAD loaded")
            except Exception as e:
                print(f"⚠️ Could not load Silero VAD: {e}")
        else:
            print("⚠️ Silero VAD not available. Using energy-based VAD.")
    
    def preprocess(self, source: Any) -> np.ndarray:
        """
        Preprocess audio input.
        
        Args:
            source: Audio samples or path
            
        Returns:
            Normalized audio samples
        """
        if isinstance(source, np.ndarray):
            audio = source
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
        
        # Normalize to float32 [-1, 1]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        return audio.astype(np.float32)
    
    def inference(self, audio: np.ndarray) -> dict:
        """
        Run audio analysis.
        
        Args:
            audio: Preprocessed audio samples
            
        Returns:
            Raw analysis results
        """
        results = {
            "audio_level_db": self._calculate_db(audio),
            "noise_level": self._estimate_noise_level(audio),
        }
        
        # Voice Activity Detection
        if self.vad_model is not None:
            vad_result = self._detect_voice(audio)
            results["voice_detected"] = vad_result["detected"]
            results["voice_confidence"] = vad_result["confidence"]
        else:
            # Energy-based fallback
            results["voice_detected"] = results["audio_level_db"] > self.cfg.silence_threshold_db
            results["voice_confidence"] = min(1.0, (results["audio_level_db"] + 60) / 40)
        
        # Whispering detection
        if results["voice_detected"]:
            results["whispering"] = self._detect_whispering(audio, results["audio_level_db"])
        else:
            results["whispering"] = False
        
        # Add to buffer for speaker analysis
        self.audio_buffer.extend(audio)
        
        # Speaker analysis (need enough buffer)
        if len(self.audio_buffer) >= self.cfg.sample_rate:
            speaker_result = self._analyze_speakers(np.array(self.audio_buffer))
            results.update(speaker_result)
        else:
            results["speaker_count"] = 1
            results["multiple_speakers"] = False
            results["speaker_change_detected"] = False
            results["background_voice"] = False
        
        # Suspicious sounds
        results["suspicious_sounds"] = self._detect_suspicious_sounds(audio)
        
        return results
    
    def postprocess(self, preds: dict, source: Any) -> AudioResults:
        """
        Postprocess analysis results.
        
        Args:
            preds: Raw analysis results
            source: Original input
            
        Returns:
            AudioResults
        """
        # Calibrate baseline if needed
        if self.calibrating:
            self._calibrate_baseline(preds)
        
        # Determine alerts
        should_alert, alert_type, severity = self._check_alert(preds)
        
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
            is_silent=preds.get("audio_level_db", -60.0) < self.cfg.silence_threshold_db,
            noise_level=preds.get("noise_level", 0.0),
            should_alert=should_alert,
            alert_type=alert_type,
            alert_severity=severity,
        )
    
    def _calculate_db(self, audio: np.ndarray) -> float:
        """Calculate audio level in decibels."""
        rms = np.sqrt(np.mean(audio ** 2))
        return float(20 * np.log10(rms)) if rms > 0 else -100
    
    def _estimate_noise_level(self, audio: np.ndarray) -> float:
        """Estimate background noise level."""
        window_size = len(audio) // 10
        if window_size < 100:
            return 0.0
        
        windows = [audio[i:i+window_size] for i in range(0, len(audio) - window_size, window_size)]
        rms_values = [np.sqrt(np.mean(w ** 2)) for w in windows]
        
        return float(np.percentile(rms_values, 10))
    
    def _detect_voice(self, audio: np.ndarray) -> dict:
        """Detect voice activity using Silero VAD."""
        if self.vad_model is None:
            return {"detected": False, "confidence": 0.0}
        
        try:
            _import_torch()
            audio_tensor = torch.from_numpy(audio).float()
            speech_prob = self.vad_model(audio_tensor, self.cfg.sample_rate).item()
            
            return {
                "detected": speech_prob > self.cfg.vad_threshold,
                "confidence": float(speech_prob),
            }
        except Exception:
            return {"detected": False, "confidence": 0.0}
    
    def _detect_whispering(self, audio: np.ndarray, db_level: float) -> bool:
        """Detect if speech is whispering."""
        if db_level > self.cfg.normal_speech_db:
            return False  # Too loud
        if db_level < self.cfg.silence_threshold_db:
            return False  # Too quiet
        
        # Check spectral characteristics
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/self.cfg.sample_rate)
        
        # Low frequency energy (100-500 Hz)
        low_mask = (freqs >= 100) & (freqs <= 500)
        low_energy = np.sum(fft[low_mask] ** 2)
        
        # High frequency energy (2000-4000 Hz)
        high_mask = (freqs >= 2000) & (freqs <= 4000)
        high_energy = np.sum(fft[high_mask] ** 2)
        
        if low_energy > 0:
            high_low_ratio = high_energy / low_energy
            return high_low_ratio > 2.0 and db_level < self.cfg.whisper_threshold_db
        
        return False
    
    def _analyze_speakers(self, audio: np.ndarray) -> dict:
        """Analyze speakers in audio segment."""
        result = {
            "speaker_count": 1,
            "multiple_speakers": False,
            "speaker_change_detected": False,
            "background_voice": False,
        }
        
        # Extract speaker features
        features = self._extract_speaker_features(audio)
        
        if self.primary_speaker_embedding is None:
            self.primary_speaker_embedding = features
            return result
        
        # Compare to primary speaker
        similarity = self._cosine_similarity(features, self.primary_speaker_embedding)
        
        if similarity < 0.7:
            result["speaker_change_detected"] = True
            self.speaker_change_count += 1
            
            if self.speaker_change_count >= 3:
                result["multiple_speakers"] = True
                result["speaker_count"] = 2
        
        # Check for overlapping speech
        if self._detect_overlap(audio):
            result["background_voice"] = True
            result["multiple_speakers"] = True
        
        return result
    
    def _extract_speaker_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC-like speaker features."""
        n_fft = 512
        hop_length = 256
        
        spectrogram = []
        for i in range(0, len(audio) - n_fft, hop_length):
            frame = audio[i:i+n_fft]
            fft = np.abs(np.fft.rfft(frame * np.hanning(n_fft)))
            spectrogram.append(fft)
        
        if not spectrogram:
            return np.zeros(13)
        
        spectrogram = np.array(spectrogram)
        
        # Simplified mel filterbank
        n_mels = 26
        mel_matrix = self._mel_filterbank(n_fft // 2 + 1, n_mels, self.cfg.sample_rate)
        mel_spec = np.dot(spectrogram, mel_matrix.T)
        
        log_mel = np.log(mel_spec + 1e-10)
        mean_log_mel = np.mean(log_mel, axis=0)
        
        # DCT for MFCC
        mfcc = np.zeros(13)
        for i in range(13):
            for j in range(n_mels):
                mfcc[i] += mean_log_mel[j] * np.cos(np.pi * i * (j + 0.5) / n_mels)
        
        return mfcc
    
    def _mel_filterbank(self, n_fft: int, n_mels: int, sample_rate: int) -> np.ndarray:
        """Create mel filterbank matrix."""
        low_mel = 2595 * np.log10(1 + 0 / 700)
        high_mel = 2595 * np.log10(1 + sample_rate / 2 / 700)
        
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((n_fft) * hz_points / sample_rate).astype(int)
        
        filterbank = np.zeros((n_mels, n_fft))
        for i in range(1, n_mels + 1):
            for j in range(bin_points[i-1], bin_points[i]):
                filterbank[i-1, j] = (j - bin_points[i-1]) / (bin_points[i] - bin_points[i-1])
            for j in range(bin_points[i], min(bin_points[i+1], n_fft)):
                filterbank[i-1, j] = (bin_points[i+1] - j) / (bin_points[i+1] - bin_points[i])
        
        return filterbank
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def _detect_overlap(self, audio: np.ndarray) -> bool:
        """Detect overlapping speech."""
        fft = np.abs(np.fft.rfft(audio))
        
        geometric_mean = np.exp(np.mean(np.log(fft + 1e-10)))
        arithmetic_mean = np.mean(fft)
        
        if arithmetic_mean > 0:
            flatness = geometric_mean / arithmetic_mean
            return flatness < 0.1
        
        return False
    
    def _detect_suspicious_sounds(self, audio: np.ndarray) -> list:
        """Detect suspicious sounds."""
        sounds = []
        
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/self.cfg.sample_rate)
        
        # High-frequency tones (notifications)
        high_freq_mask = (freqs >= 1000) & (freqs <= 4000)
        high_freq_energy = np.sum(fft[high_freq_mask] ** 2)
        total_energy = np.sum(fft ** 2)
        
        if total_energy > 0:
            high_freq_ratio = high_freq_energy / total_energy
            if high_freq_ratio > 0.8:
                sounds.append("possible_notification")
        
        # Typing sounds
        envelope = np.abs(audio)
        diff = np.diff(envelope)
        transient_count = np.sum(np.abs(diff) > 0.1)
        
        if transient_count > len(audio) / 1000:
            sounds.append("possible_typing")
        
        return sounds
    
    def _calibrate_baseline(self, result: dict):
        """Calibrate baseline noise level."""
        self.calibration_samples.append(result.get("noise_level", 0))
        
        if len(self.calibration_samples) >= self.cfg.calibration_samples:
            self.baseline_noise_level = np.median(self.calibration_samples)
            self.calibrating = False
            print(f"✅ Audio baseline calibrated: noise_level={self.baseline_noise_level:.2f}")
    
    def _check_alert(self, result: dict) -> tuple[bool, Optional[str], str]:
        """Check if alert should be triggered."""
        if result.get("multiple_speakers"):
            return True, "multiple_voices", "high"
        
        if result.get("background_voice"):
            return True, "background_voice", "high"
        
        if result.get("whispering"):
            return True, "whispering", "medium"
        
        if "possible_notification" in result.get("suspicious_sounds", []):
            return True, "phone_sound", "medium"
        
        return False, None, "low"
    
    def calibrate(self, audio_samples: np.ndarray, sample_rate: int = 16000):
        """Manually calibrate baseline."""
        audio = self.preprocess(audio_samples)
        noise_level = self._estimate_noise_level(audio)
        self.baseline_noise_level = noise_level
        self.calibrating = False
        print(f"✅ Audio baseline set: noise_level={noise_level:.2f}")
    
    def reset(self):
        """Reset predictor state."""
        self.audio_buffer.clear()
        self.primary_speaker_embedding = None
        self.speaker_change_count = 0
        self.total_voice_duration = 0
        self.total_silence_duration = 0
        self.baseline_noise_level = None
        self.calibrating = True
        self.calibration_samples.clear()
    
    def get_statistics(self) -> dict:
        """Get audio analysis statistics."""
        total_duration = self.total_voice_duration + self.total_silence_duration
        
        return {
            "total_voice_duration_ms": self.total_voice_duration,
            "total_silence_duration_ms": self.total_silence_duration,
            "voice_ratio": self.total_voice_duration / total_duration if total_duration > 0 else 0,
            "speaker_changes": self.speaker_change_count,
            "baseline_noise_level": self.baseline_noise_level,
        }
