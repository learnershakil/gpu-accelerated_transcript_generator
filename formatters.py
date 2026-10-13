"""Formatters and exporters for transcripts in multiple formats:
- .txt  : Clean transcript with or without timestamps
- .srt  : Standard SubRip subtitle format
- .vtt  : WebVTT standard subtitle format
- .json : Detailed structured JSON with metadata and word-by-word timestamps
- .words.txt : Granular word-level timestamp breakdown
- .md   : Formatted markdown document
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def format_timestamp(seconds: float, *, srt: bool = False, vtt: bool = False) -> str:
    """Convert float seconds to timestamp string.
    
    SRT format: HH:MM:SS,mmm
    VTT format: HH:MM:SS.mmm
    Standard:   HH:MM:SS.mmm
    """
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    separator = "," if srt else "."
    return f"{hours:02}:{minutes:02}:{whole_seconds:02}{separator}{millis:03}"


def export_srt(segments: List[Dict[str, Any]], output_path: Path) -> Path:
    """Generate SubRip (.srt) subtitle file."""
    lines: List[str] = []
    for idx, seg in enumerate(segments, start=1):
        start_ts = format_timestamp(seg["start"], srt=True)
        end_ts = format_timestamp(seg["end"], srt=True)
        text = seg["text"].strip()
        lines.append(f"{idx}\n{start_ts} --> {end_ts}\n{text}\n")
    
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def export_vtt(segments: List[Dict[str, Any]], output_path: Path) -> Path:
    """Generate WebVTT (.vtt) subtitle file."""
    lines: List[str] = ["WEBVTT\n"]
    for idx, seg in enumerate(segments, start=1):
        start_ts = format_timestamp(seg["start"], vtt=True)
        end_ts = format_timestamp(seg["end"], vtt=True)
        text = seg["text"].strip()
        lines.append(f"{idx}\n{start_ts} --> {end_ts}\n{text}\n")
    
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def export_txt(
    segments: List[Dict[str, Any]],
    output_path: Path,
    *,
    include_timestamps: bool = True,
) -> Path:
    """Generate clean plain text (.txt) transcript."""
    lines: List[str] = []
    for seg in segments:
        text = seg["text"].strip()
        if include_timestamps:
            start_ts = format_timestamp(seg["start"])
            end_ts = format_timestamp(seg["end"])
            lines.append(f"[{start_ts} --> {end_ts}]  {text}")
        else:
            lines.append(text)
    
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def export_words(segments: List[Dict[str, Any]], output_path: Path) -> Path:
    """Generate detailed word-by-word timestamp log (.words.txt)."""
    lines: List[str] = [
        "# Word-level Timestamp Breakdown",
        "# [START_TIME --> END_TIME] (CONFIDENCE) WORD",
        "-" * 60,
    ]
    for seg in segments:
        words = seg.get("words", [])
        if not words:
            continue
        for w in words:
            start_ts = format_timestamp(w.get("start", 0.0))
            end_ts = format_timestamp(w.get("end", 0.0))
            prob = w.get("probability", 1.0)
            conf_str = f"({prob * 100:.1f}%)" if prob is not None else "(100%)"
            word_str = w.get("word", "").strip()
            lines.append(f"[{start_ts} --> {end_ts}] {conf_str:<8} {word_str}")
        lines.append("")  # Blank line between segments
    
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def export_json(
    segments: List[Dict[str, Any]],
    info_dict: Dict[str, Any],
    metadata: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Generate complete structured JSON (.json) with all data."""
    data = {
        "metadata": metadata,
        "transcription_info": info_dict,
        "segment_count": len(segments),
        "total_words": sum(len(seg.get("words", [])) for seg in segments),
        "segments": segments,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return output_path


def export_markdown(
    segments: List[Dict[str, Any]],
    info_dict: Dict[str, Any],
    metadata: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Generate beautifully formatted Markdown document (.md)."""
    lang = info_dict.get("language", "unknown").upper()
    lang_prob = info_dict.get("language_probability", 0.0)
    duration = info_dict.get("duration", 0.0)
    dur_str = format_timestamp(duration)
    filename = metadata.get("source_filename", "Audio/Video File")
    model = metadata.get("model", "Whisper")
    device = metadata.get("device", "GPU")

    lines: List[str] = [
        f"# Transcription: {filename}",
        "",
        "## Summary & Metadata",
        f"- **Source File**: `{filename}`",
        f"- **Duration**: `{dur_str}` ({duration:.2f}s)",
        f"- **Language Detected**: `{lang}` ({lang_prob * 100:.1f}% confidence)",
        f"- **Model Used**: `{model}`",
        f"- **Hardware Acceleration**: `{device}`",
        f"- **Total Segments**: `{len(segments)}`",
        "",
        "---",
        "",
        "## Transcript",
        "",
    ]

    for seg in segments:
        start_ts = format_timestamp(seg["start"])
        end_ts = format_timestamp(seg["end"])
        text = seg["text"].strip()
        lines.append(f"> **`{start_ts} - {end_ts}`**  \n> {text}\n")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
