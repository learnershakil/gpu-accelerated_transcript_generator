#!/usr/bin/env python3
"""Legacy and CLI entry point for the Transcript Generator.
Can be executed directly:
    python transcribe_audio.py [optional_path_to_audio_or_video]
Or run interactively:
    python main.py  (or ./run.sh)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from gpu_utils import setup_cuda_environment
setup_cuda_environment()

import main


def cli() -> int:
    # If a file path was passed via command line argument, process it directly
    if len(sys.argv) > 1 and sys.argv[1] not in ("-h", "--help"):
        target_path = Path(sys.argv[1]).expanduser().resolve()
        if not target_path.is_file():
            print(f"Error: Specified file does not exist: {target_path}", file=sys.stderr)
            return 1
        
        gpu_info = main.get_gpu_info()
        main.display_banner(gpu_info)
        mode = main.select_transcription_mode()
        config = main.configure_transcription(target_path, gpu_info, mode)
        
        print("\nLoading model and initiating transcription...")
        engine = main.TranscriptionEngine(
            model_size=config["model_size"],
            device=config["device"],
            compute_type=config["compute_type"],
        )
        
        segments, info_dict = engine.transcribe(
            config["file_path"],
            mode=config["mode"],
            language=config["language"],
            vad_filter=config["vad_filter"],
            beam_size=config["beam_size"],
            word_timestamps=True,
        )
        
        out_dir = config["output_dir"]
        base = target_path.stem
        main.export_txt(segments, out_dir / f"{base}_transcript.txt")
        main.export_srt(segments, out_dir / f"{base}_transcript.srt")
        main.export_vtt(segments, out_dir / f"{base}_transcript.vtt")
        main.export_words(segments, out_dir / f"{base}_words.txt")
        metadata = {
            "source_filepath": str(target_path),
            "source_filename": target_path.name,
            "model": config["model_size"],
            "device": engine.actual_device,
            "compute_type": engine.actual_compute_type,
        }
        main.export_json(segments, info_dict, metadata, out_dir / f"{base}_transcript.json")
        main.export_markdown(segments, info_dict, metadata, out_dir / f"{base}_transcript.md")
        
        print(f"\n✔ Complete! Transcripts saved to {out_dir}")
        return 0

    # Otherwise run the full interactive guided pipeline
    return main.run_pipeline()


if __name__ == "__main__":
    sys.exit(cli())
