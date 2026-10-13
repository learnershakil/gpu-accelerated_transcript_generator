"""Configuration and format definitions for the transcript generator pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

# Base Project Directories
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "transcripts"

# Top 10 Video Formats
VIDEO_FORMATS: List[Tuple[str, str, str]] = [
    ("MP4", ".mp4", "MPEG-4 Video (Universal standard)"),
    ("MKV", ".mkv", "Matroska Video (Supports multiple tracks/subs)"),
    ("MOV", ".mov", "Apple QuickTime Movie"),
    ("AVI", ".avi", "Audio Video Interleave (Legacy standard)"),
    ("WEBM", ".webm", "WebM Video (HTML5 web media)"),
    ("FLV", ".flv", "Flash Video Stream"),
    ("WMV", ".wmv", "Windows Media Video"),
    ("M4V", ".m4v", "Apple iTunes Video container"),
    ("TS", ".ts", "MPEG Transport Stream (Broadcast)"),
    ("3GP", ".3gp", "3GPP Mobile Multimedia"),
]

# Top 10 Audio Formats
AUDIO_FORMATS: List[Tuple[str, str, str]] = [
    ("MP3", ".mp3", "MPEG Audio Layer III (Most popular)"),
    ("WAV", ".wav", "Waveform Audio File Format (Lossless PCM)"),
    ("M4A", ".m4a", "MPEG-4 Audio (High quality AAC/ALAC)"),
    ("FLAC", ".flac", "Free Lossless Audio Codec (Studio quality)"),
    ("AAC", ".aac", "Advanced Audio Coding"),
    ("OGG", ".ogg", "Ogg Vorbis Audio"),
    ("WMA", ".wma", "Windows Media Audio"),
    ("OPUS", ".opus", "Opus Interactive Audio Codec"),
    ("AIFF", ".aiff", "Audio Interchange File Format (Apple PCM)"),
    ("AMR", ".amr", "Adaptive Multi-Rate Audio (Voice recordings)"),
]

# Available Whisper Models
WHISPER_MODELS: Dict[str, Dict[str, str]] = {
    "large-v3": {
        "name": "large-v3",
        "description": "State-of-the-art multilingual model (Highest accuracy & completeness)",
        "vram": "~3.5 - 4.5 GB",
        "recommended": True,
    },
    "large-v3-turbo": {
        "name": "large-v3-turbo",
        "description": "Optimized large model (8x faster, ultra-low VRAM ~2.5 GB)",
        "vram": "~2.5 GB",
        "recommended": False,
    },
    "medium": {
        "name": "medium",
        "description": "Great balance for multilingual content (~1.5 GB VRAM)",
        "vram": "~1.5 GB",
        "recommended": False,
    },
    "small": {
        "name": "small",
        "description": "Lightweight model (~1.0 GB VRAM)",
        "vram": "~1.0 GB",
        "recommended": False,
    },
    "base": {
        "name": "base",
        "description": "Fast prototype model (~0.5 GB VRAM)",
        "vram": "~0.5 GB",
        "recommended": False,
    },
}

# GPU / Hardware Configuration
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
DEFAULT_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
DEFAULT_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
DEFAULT_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))
DEFAULT_WORD_TIMESTAMPS = True
DEFAULT_VAD_FILTER = os.environ.get("WHISPER_VAD", "false").lower() in ("true", "1", "yes")

# Target GPU Spec
TARGET_GPU_NAME = "NVIDIA GeForce RTX 3050"
