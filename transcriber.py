"""Core transcription engine using Faster-Whisper with GPU acceleration."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from faster_whisper import WhisperModel

from gpu_utils import get_gpu_info, setup_cuda_environment


class TranscriptionEngine:
    """Wrapper around faster-whisper WhisperModel with hardware detection,
    live streaming callbacks, and graceful fallback.
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        cpu_threads: Optional[int] = None,
        num_workers: int = 1,
    ) -> None:
        self.model_size = model_size
        self.requested_device = device
        self.requested_compute_type = compute_type
        self.cpu_threads = cpu_threads or max(1, os.cpu_count() or 4)
        self.num_workers = num_workers
        self.actual_device = device
        self.actual_compute_type = compute_type
        self.model: Optional[WhisperModel] = None

        # Pre-configure CUDA environment and shared libraries
        setup_cuda_environment()
        self._load_model()

    def _load_model(self) -> None:
        """Load model onto device with fallback handling."""
        try:
            if self.requested_device == "cuda":
                # Ensure CUDA is accessible
                gpu = get_gpu_info()
                if not gpu["cuda_available"]:
                    print(
                        f"⚠️  CUDA not ready ({gpu.get('driver_warning')}). Falling back to CPU.",
                        file=sys.stderr,
                    )
                    self._fallback_to_cpu()
                    return

            self.model = WhisperModel(
                self.model_size,
                device=self.requested_device,
                compute_type=self.requested_compute_type,
                cpu_threads=self.cpu_threads,
                num_workers=self.num_workers,
            )
            self.actual_device = self.requested_device
            self.actual_compute_type = self.requested_compute_type
        except Exception as e:
            err_msg = str(e)
            if self.requested_device == "cuda":
                print(
                    f"⚠️  GPU load failed: {err_msg}\nFalling back to CPU (int8)...",
                    file=sys.stderr,
                )
                self._fallback_to_cpu()
            else:
                raise e

    def _fallback_to_cpu(self) -> None:
        self.actual_device = "cpu"
        self.actual_compute_type = "int8"
        self.model = WhisperModel(
            self.model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=self.cpu_threads,
            num_workers=self.num_workers,
        )

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: Optional[str] = None,
        vad_filter: bool = False,
        beam_size: int = 5,
        word_timestamps: bool = True,
        on_segment: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_progress: Optional[Callable[[float, float, float], None]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Transcribe an audio or video file with real-time segment streaming.
        
        Args:
            audio_path: Path to audio or video media file.
            language: Optional language code (e.g. 'en', 'hi') or None for auto-detect.
            vad_filter: Whether to apply Silero VAD to filter non-speech.
            beam_size: Beam search width (5 gives optimal correctness).
            word_timestamps: Extract word-level start/end timestamps.
            on_segment: Callback invoked for each new segment.
            on_progress: Callback invoked with (progress_pct, current_sec, total_sec).
            
        Returns:
            Tuple of (segments_list, info_dict).
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        audio_path_str = str(Path(audio_path).resolve())
        start_time = time.time()

        segments_iter, transcription_info = self.model.transcribe(
            audio_path_str,
            language=language,
            task="transcribe",
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
        )

        total_duration = max(0.1, float(getattr(transcription_info, "duration", 0.0) or 0.1))
        info_dict: Dict[str, Any] = {
            "language": getattr(transcription_info, "language", "unknown"),
            "language_probability": round(float(getattr(transcription_info, "language_probability", 1.0) or 1.0), 4),
            "duration": total_duration,
            "duration_after_vad": round(float(getattr(transcription_info, "duration_after_vad", total_duration) or total_duration), 2),
            "all_language_probs": getattr(transcription_info, "all_language_probs", None),
        }

        collected_segments: List[Dict[str, Any]] = []

        for seg in segments_iter:
            seg_dict: Dict[str, Any] = {
                "id": seg.id,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text,
                "words": [],
            }

            if seg.words:
                for w in seg.words:
                    seg_dict["words"].append({
                        "word": w.word,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 3) if w.probability is not None else 1.0,
                    })

            collected_segments.append(seg_dict)

            if on_segment:
                on_segment(seg_dict)

            if on_progress:
                progress_pct = min(100.0, (seg.end / total_duration) * 100.0)
                on_progress(progress_pct, seg.end, total_duration)

        elapsed = time.time() - start_time
        info_dict["processing_time_seconds"] = round(elapsed, 2)
        info_dict["speed_factor"] = round(total_duration / elapsed, 2) if elapsed > 0 else 0.0

        return collected_segments, info_dict
