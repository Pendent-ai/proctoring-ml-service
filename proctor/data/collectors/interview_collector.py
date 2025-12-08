"""
Interview Data Collection Pipeline

Collects and labels video/audio data for training proctoring models.
Supports both manual annotation and semi-automated labeling.
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import numpy as np


class BehaviorLabel(Enum):
    """Video behavior labels for proctoring"""
    NORMAL = "normal"
    LOOKING_AWAY = "looking_away"
    LOOKING_DOWN = "looking_down"
    READING_NOTES = "reading_notes"
    USING_PHONE = "using_phone"
    TALKING_TO_SOMEONE = "talking_to_someone"
    TYPING = "typing"
    SUSPICIOUS_MOVEMENT = "suspicious_movement"
    LEFT_FRAME = "left_frame"
    MULTIPLE_FACES = "multiple_faces"


class AudioLabel(Enum):
    """Audio labels for proctoring"""
    NORMAL_SPEECH = "normal_speech"
    SILENCE = "silence"
    MULTIPLE_VOICES = "multiple_voices"
    WHISPER = "whisper"
    BACKGROUND_SPEECH = "background_speech"
    AI_GENERATED = "ai_generated"
    KEYBOARD_TYPING = "keyboard_typing"
    PHONE_NOTIFICATION = "phone_notification"


@dataclass
class TimeSegment:
    """A labeled time segment in video/audio"""
    start_time: float  # seconds
    end_time: float
    label: str
    confidence: float = 1.0
    annotator: str = "manual"
    notes: str = ""


@dataclass
class VideoAnnotation:
    """Annotation for a video clip"""
    video_id: str
    video_path: str
    duration: float
    fps: float
    resolution: Tuple[int, int]
    segments: List[TimeSegment] = field(default_factory=list)
    frame_annotations: Dict[int, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass  
class AudioAnnotation:
    """Annotation for an audio clip"""
    audio_id: str
    audio_path: str
    duration: float
    sample_rate: int
    segments: List[TimeSegment] = field(default_factory=list)
    transcript: Optional[str] = None
    speaker_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class InterviewDataCollector:
    """
    Collects interview recordings and manages annotations.
    
    Workflow:
    1. Record or import interview videos
    2. Extract audio tracks
    3. Run pre-annotation with existing models
    4. Manual review and correction
    5. Export for training
    """
    
    def __init__(self, data_dir: str = "data/interviews"):
        self.data_dir = Path(data_dir)
        self.videos_dir = self.data_dir / "videos"
        self.audio_dir = self.data_dir / "audio"
        self.annotations_dir = self.data_dir / "annotations"
        self.exports_dir = self.data_dir / "exports"
        
        # Create directories
        for d in [self.videos_dir, self.audio_dir, self.annotations_dir, self.exports_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.annotations: Dict[str, VideoAnnotation] = {}
        self.audio_annotations: Dict[str, AudioAnnotation] = {}
    
    def import_video(
        self,
        video_path: str,
        extract_audio: bool = True,
        run_pre_annotation: bool = True
    ) -> str:
        """Import a video file for annotation"""
        import cv2
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Generate unique ID
        with open(video_path, 'rb') as f:
            video_id = hashlib.md5(f.read(1024*1024)).hexdigest()[:12]
        
        # Get video properties
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # Copy to data directory
        dest_path = self.videos_dir / f"{video_id}.mp4"
        if not dest_path.exists():
            import shutil
            shutil.copy(video_path, dest_path)
        
        # Create annotation object
        annotation = VideoAnnotation(
            video_id=video_id,
            video_path=str(dest_path),
            duration=duration,
            fps=fps,
            resolution=(width, height),
            metadata={
                "original_path": str(video_path),
                "frame_count": frame_count
            }
        )
        
        self.annotations[video_id] = annotation
        self._save_annotation(annotation)
        
        # Extract audio
        if extract_audio:
            self._extract_audio(video_id, dest_path)
        
        # Run pre-annotation
        if run_pre_annotation:
            self._pre_annotate_video(video_id)
        
        return video_id
    
    def _extract_audio(self, video_id: str, video_path: Path) -> str:
        """Extract audio from video"""
        import subprocess
        
        audio_path = self.audio_dir / f"{video_id}.wav"
        
        if not audio_path.exists():
            cmd = [
                "ffmpeg", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                str(audio_path), "-y"
            ]
            subprocess.run(cmd, capture_output=True)
        
        # Create audio annotation
        import wave
        with wave.open(str(audio_path), 'rb') as wf:
            sample_rate = wf.getframerate()
            duration = wf.getnframes() / sample_rate
        
        audio_annotation = AudioAnnotation(
            audio_id=video_id,
            audio_path=str(audio_path),
            duration=duration,
            sample_rate=sample_rate
        )
        
        self.audio_annotations[video_id] = audio_annotation
        return str(audio_path)
    
    def _pre_annotate_video(self, video_id: str):
        """
        Run existing models to pre-annotate video.
        This provides initial labels that humans can review/correct.
        """
        annotation = self.annotations[video_id]
        
        # Would use existing proctoring pipeline here
        # For now, create placeholder segments
        segment_duration = 5.0  # 5-second segments
        
        segments = []
        current_time = 0.0
        while current_time < annotation.duration:
            end_time = min(current_time + segment_duration, annotation.duration)
            segments.append(TimeSegment(
                start_time=current_time,
                end_time=end_time,
                label=BehaviorLabel.NORMAL.value,
                confidence=0.5,  # Low confidence = needs review
                annotator="auto"
            ))
            current_time = end_time
        
        annotation.segments = segments
        self._save_annotation(annotation)
    
    def add_segment_label(
        self,
        video_id: str,
        start_time: float,
        end_time: float,
        label: BehaviorLabel,
        annotator: str = "manual",
        notes: str = ""
    ):
        """Add or update a labeled segment"""
        if video_id not in self.annotations:
            raise ValueError(f"Video {video_id} not found")
        
        annotation = self.annotations[video_id]
        
        # Remove overlapping segments
        annotation.segments = [
            s for s in annotation.segments
            if s.end_time <= start_time or s.start_time >= end_time
        ]
        
        # Add new segment
        annotation.segments.append(TimeSegment(
            start_time=start_time,
            end_time=end_time,
            label=label.value,
            confidence=1.0,
            annotator=annotator,
            notes=notes
        ))
        
        # Sort by time
        annotation.segments.sort(key=lambda s: s.start_time)
        self._save_annotation(annotation)
    
    def add_audio_label(
        self,
        audio_id: str,
        start_time: float,
        end_time: float,
        label: AudioLabel,
        annotator: str = "manual"
    ):
        """Add audio segment label"""
        if audio_id not in self.audio_annotations:
            raise ValueError(f"Audio {audio_id} not found")
        
        annotation = self.audio_annotations[audio_id]
        
        annotation.segments.append(TimeSegment(
            start_time=start_time,
            end_time=end_time,
            label=label.value,
            confidence=1.0,
            annotator=annotator
        ))
        
        annotation.segments.sort(key=lambda s: s.start_time)
        self._save_audio_annotation(annotation)
    
    def _save_annotation(self, annotation: VideoAnnotation):
        """Save annotation to disk"""
        path = self.annotations_dir / f"{annotation.video_id}_video.json"
        with open(path, 'w') as f:
            data = asdict(annotation)
            json.dump(data, f, indent=2)
    
    def _save_audio_annotation(self, annotation: AudioAnnotation):
        """Save audio annotation to disk"""
        path = self.annotations_dir / f"{annotation.audio_id}_audio.json"
        with open(path, 'w') as f:
            data = asdict(annotation)
            json.dump(data, f, indent=2)
    
    def load_annotations(self):
        """Load all annotations from disk"""
        for path in self.annotations_dir.glob("*_video.json"):
            with open(path) as f:
                data = json.load(f)
                data['segments'] = [TimeSegment(**s) for s in data['segments']]
                data['resolution'] = tuple(data['resolution'])
                self.annotations[data['video_id']] = VideoAnnotation(**data)
        
        for path in self.annotations_dir.glob("*_audio.json"):
            with open(path) as f:
                data = json.load(f)
                data['segments'] = [TimeSegment(**s) for s in data['segments']]
                self.audio_annotations[data['audio_id']] = AudioAnnotation(**data)
    
    def export_for_training(
        self,
        output_dir: Optional[str] = None,
        clip_duration: float = 3.0,
        overlap: float = 0.5
    ) -> Dict[str, str]:
        """
        Export labeled clips for training.
        
        Creates:
        - video_clips/ - Short video clips
        - audio_clips/ - Audio segments
        - labels.json - All labels
        """
        import cv2
        import subprocess
        
        output_dir = Path(output_dir or self.exports_dir / datetime.now().strftime("%Y%m%d_%H%M%S"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_clips_dir = output_dir / "video_clips"
        audio_clips_dir = output_dir / "audio_clips"
        video_clips_dir.mkdir(exist_ok=True)
        audio_clips_dir.mkdir(exist_ok=True)
        
        all_labels = {
            "video_clips": [],
            "audio_clips": [],
            "label_mapping": {
                "video": {l.value: i for i, l in enumerate(BehaviorLabel)},
                "audio": {l.value: i for i, l in enumerate(AudioLabel)}
            }
        }
        
        # Export video clips
        for video_id, annotation in self.annotations.items():
            cap = cv2.VideoCapture(annotation.video_path)
            fps = annotation.fps
            
            for segment in annotation.segments:
                if segment.confidence < 0.8:  # Skip low-confidence
                    continue
                
                # Extract clip
                clip_id = f"{video_id}_{int(segment.start_time*1000)}"
                clip_path = video_clips_dir / f"{clip_id}.mp4"
                
                start_frame = int(segment.start_time * fps)
                end_frame = int(segment.end_time * fps)
                
                # Use ffmpeg for precise cutting
                cmd = [
                    "ffmpeg", "-i", annotation.video_path,
                    "-ss", str(segment.start_time),
                    "-t", str(segment.end_time - segment.start_time),
                    "-c:v", "libx264", "-c:a", "aac",
                    str(clip_path), "-y"
                ]
                subprocess.run(cmd, capture_output=True)
                
                all_labels["video_clips"].append({
                    "clip_id": clip_id,
                    "path": str(clip_path),
                    "label": segment.label,
                    "label_id": all_labels["label_mapping"]["video"].get(segment.label, -1),
                    "duration": segment.end_time - segment.start_time,
                    "source_video": video_id
                })
            
            cap.release()
        
        # Export audio clips
        for audio_id, annotation in self.audio_annotations.items():
            for segment in annotation.segments:
                if segment.confidence < 0.8:
                    continue
                
                clip_id = f"{audio_id}_{int(segment.start_time*1000)}"
                clip_path = audio_clips_dir / f"{clip_id}.wav"
                
                cmd = [
                    "ffmpeg", "-i", annotation.audio_path,
                    "-ss", str(segment.start_time),
                    "-t", str(segment.end_time - segment.start_time),
                    str(clip_path), "-y"
                ]
                subprocess.run(cmd, capture_output=True)
                
                all_labels["audio_clips"].append({
                    "clip_id": clip_id,
                    "path": str(clip_path),
                    "label": segment.label,
                    "label_id": all_labels["label_mapping"]["audio"].get(segment.label, -1),
                    "duration": segment.end_time - segment.start_time,
                    "source_audio": audio_id
                })
        
        # Save labels
        labels_path = output_dir / "labels.json"
        with open(labels_path, 'w') as f:
            json.dump(all_labels, f, indent=2)
        
        print(f"Exported {len(all_labels['video_clips'])} video clips")
        print(f"Exported {len(all_labels['audio_clips'])} audio clips")
        print(f"Labels saved to {labels_path}")
        
        return {
            "output_dir": str(output_dir),
            "labels_path": str(labels_path),
            "video_clips": len(all_labels['video_clips']),
            "audio_clips": len(all_labels['audio_clips'])
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        video_label_counts = {}
        audio_label_counts = {}
        total_video_duration = 0
        total_audio_duration = 0
        
        for annotation in self.annotations.values():
            total_video_duration += annotation.duration
            for segment in annotation.segments:
                video_label_counts[segment.label] = video_label_counts.get(segment.label, 0) + 1
        
        for annotation in self.audio_annotations.values():
            total_audio_duration += annotation.duration
            for segment in annotation.segments:
                audio_label_counts[segment.label] = audio_label_counts.get(segment.label, 0) + 1
        
        return {
            "total_videos": len(self.annotations),
            "total_audio": len(self.audio_annotations),
            "total_video_duration_hours": total_video_duration / 3600,
            "total_audio_duration_hours": total_audio_duration / 3600,
            "video_label_counts": video_label_counts,
            "audio_label_counts": audio_label_counts
        }


class SyntheticDataGenerator:
    """
    Generate synthetic training data by augmenting real interviews.
    
    Techniques:
    - Add synthetic phone overlays
    - Mix in background voices
    - Simulate looking away behavior
    - Add AI-generated speech samples
    """
    
    def __init__(self, collector: InterviewDataCollector):
        self.collector = collector
    
    def add_phone_overlay(
        self,
        video_path: str,
        phone_image_path: str,
        position: Tuple[float, float] = (0.7, 0.6),
        duration: Tuple[float, float] = (2.0, 5.0),
        output_path: Optional[str] = None
    ) -> str:
        """
        Add a phone overlay to simulate phone usage.
        Creates training data for phone detection.
        """
        import cv2
        import random
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Load phone image
        phone_img = cv2.imread(phone_image_path, cv2.IMREAD_UNCHANGED)
        if phone_img is None:
            raise ValueError(f"Could not load phone image: {phone_image_path}")
        
        # Resize phone
        phone_scale = 0.15
        phone_h = int(height * phone_scale)
        phone_w = int(phone_h * phone_img.shape[1] / phone_img.shape[0])
        phone_img = cv2.resize(phone_img, (phone_w, phone_h))
        
        # Calculate position
        x = int(width * position[0] - phone_w // 2)
        y = int(height * position[1] - phone_h // 2)
        
        # Random appearance time
        appear_duration = random.uniform(duration[0], duration[1])
        appear_start = random.uniform(0, max(0, (total_frames / fps) - appear_duration - 1))
        appear_end = appear_start + appear_duration
        
        # Output path
        if output_path is None:
            output_path = video_path.replace(".mp4", "_with_phone.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        annotations = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_idx / fps
            
            if appear_start <= current_time <= appear_end:
                # Overlay phone
                if phone_img.shape[2] == 4:  # Has alpha
                    alpha = phone_img[:, :, 3] / 255.0
                    for c in range(3):
                        frame[y:y+phone_h, x:x+phone_w, c] = \
                            alpha * phone_img[:, :, c] + \
                            (1 - alpha) * frame[y:y+phone_h, x:x+phone_w, c]
                else:
                    frame[y:y+phone_h, x:x+phone_w] = phone_img
                
                # Record annotation (YOLO format)
                cx = (x + phone_w/2) / width
                cy = (y + phone_h/2) / height
                w = phone_w / width
                h = phone_h / height
                annotations.append({
                    "frame": frame_idx,
                    "class": "phone_in_hand",
                    "bbox": [cx, cy, w, h]
                })
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        
        # Save annotations
        ann_path = output_path.replace(".mp4", "_annotations.json")
        with open(ann_path, 'w') as f:
            json.dump(annotations, f)
        
        return output_path
    
    def mix_background_voice(
        self,
        audio_path: str,
        background_voice_path: str,
        mix_ratio: float = 0.3,
        output_path: Optional[str] = None
    ) -> str:
        """
        Mix in background voice to simulate multiple speakers.
        Creates training data for voice detection.
        """
        import wave
        import numpy as np
        
        # Load main audio
        with wave.open(audio_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            main_audio = np.frombuffer(wf.readframes(-1), dtype=np.int16).astype(np.float32)
        
        # Load background
        with wave.open(background_voice_path, 'rb') as wf:
            bg_audio = np.frombuffer(wf.readframes(-1), dtype=np.int16).astype(np.float32)
        
        # Match lengths
        if len(bg_audio) < len(main_audio):
            bg_audio = np.tile(bg_audio, int(np.ceil(len(main_audio) / len(bg_audio))))
        bg_audio = bg_audio[:len(main_audio)]
        
        # Mix
        mixed = main_audio * (1 - mix_ratio) + bg_audio * mix_ratio
        mixed = np.clip(mixed, -32768, 32767).astype(np.int16)
        
        # Save
        if output_path is None:
            output_path = audio_path.replace(".wav", "_mixed.wav")
        
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(mixed.tobytes())
        
        return output_path
    
    def generate_ai_speech_samples(
        self,
        texts: List[str],
        output_dir: str,
        voice: str = "default"
    ) -> List[str]:
        """
        Generate AI speech samples for training AI voice detector.
        Uses TTS to create obviously AI-generated speech.
        """
        from pathlib import Path
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # Would integrate with TTS service here
        # For now, placeholder
        for i, text in enumerate(texts):
            output_path = output_dir / f"ai_speech_{i:04d}.wav"
            # tts.synthesize(text, output_path)
            generated_files.append(str(output_path))
        
        return generated_files
