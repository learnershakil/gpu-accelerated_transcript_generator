#!/usr/bin/env python3
"""Interactive GPU-Accelerated Audio/Video Transcription Pipeline.
Optimized for NVIDIA GeForce RTX 3050 Laptop GPU with CUDA.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from config import (
    AUDIO_FORMATS,
    DEFAULT_BEAM_SIZE,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VAD_FILTER,
    DEFAULT_WORD_TIMESTAMPS,
    TARGET_GPU_NAME,
    VIDEO_FORMATS,
    WHISPER_MODELS,
)
from formatters import (
    export_json,
    export_markdown,
    export_srt,
    export_txt,
    export_vtt,
    export_words,
    format_timestamp,
)
from gpu_utils import get_gpu_info, get_recommended_settings, setup_cuda_environment
from transcriber import TranscriptionEngine

console = Console()


def display_banner(gpu_info: Dict[str, Any]) -> None:
    """Display modern CLI header banner with hardware diagnostics."""
    console.print()
    banner_text = Text()
    banner_text.append("🎙️  TRANSCRIPT GENERATOR PIPELINE\n", style="bold cyan")
    banner_text.append("⚡ High-Accuracy AI Transcription powered by Faster-Whisper & CUDA\n\n", style="italic")

    # Hardware status badge
    if gpu_info["cuda_available"]:
        gpu_name = gpu_info.get("device_name", TARGET_GPU_NAME)
        vram = gpu_info.get("total_memory_gb", 4.0)
        compute_types = ", ".join(gpu_info.get("compute_types_supported", ["float16"]))
        banner_text.append(f"🟢 GPU Detected: {gpu_name} ({vram} GB VRAM)\n", style="bold green")
        banner_text.append(f"   Acceleration : CUDA Active  |  Supported: {compute_types}\n", style="dim")
    else:
        banner_text.append("🟡 Hardware: CPU Mode (CUDA unavailable or mismatched)\n", style="bold yellow")
        if gpu_info.get("driver_warning"):
            banner_text.append(f"   Notice: {gpu_info['driver_warning']}\n", style="dim yellow")

    console.print(Panel(banner_text, border_style="cyan", padding=(1, 2)))


def select_media_type() -> str:
    """Prompt user to choose between Video and Audio."""
    console.print("\n[bold cyan]Step 1: Select Media Type[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta", border_style="dim")
    table.add_column("Option", style="bold yellow", width=8, justify="center")
    table.add_column("Type", style="bold white", width=16)
    table.add_column("Description", style="dim")

    table.add_row("1", "🎥 Video", "Movies, recorded lectures, clips (.mp4, .mkv, .mov, etc.)")
    table.add_row("2", "🎵 Audio", "Voice notes, podcasts, recordings (.mp3, .wav, .m4a, etc.)")
    console.print(table)

    while True:
        choice = Prompt.ask("[bold green]Choose type[/bold green]", choices=["1", "2", "video", "audio", "v", "a"], default="1")
        if choice in ("1", "video", "v"):
            return "video"
        elif choice in ("2", "audio", "a"):
            return "audio"


def select_format(media_type: str) -> Tuple[str, str]:
    """Prompt user to choose from the top 10 popular formats."""
    is_video = media_type == "video"
    formats = VIDEO_FORMATS if is_video else AUDIO_FORMATS
    cat_name = "Video" if is_video else "Audio"

    console.print(f"\n[bold cyan]Step 2: Select {cat_name} File Format (Top 10 Most Popular)[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta", border_style="dim")
    table.add_column("#", style="bold yellow", width=4, justify="right")
    table.add_column("Format", style="bold cyan", width=10)
    table.add_column("Extension", style="bold green", width=12)
    table.add_column("Description", style="white")

    for idx, (fmt_name, ext, desc) in enumerate(formats, start=1):
        table.add_row(str(idx), fmt_name, ext, desc)
    table.add_row(str(len(formats) + 1), "OTHER", "Custom", "Any other format or auto-detect extension")
    console.print(table)

    max_choice = len(formats) + 1
    while True:
        try:
            choice_str = Prompt.ask(f"[bold green]Select format number (1-{max_choice})[/bold green]", default="1")
            choice_idx = int(choice_str)
            if 1 <= choice_idx <= len(formats):
                selected = formats[choice_idx - 1]
                return selected[0], selected[1]
            elif choice_idx == max_choice:
                custom_ext = Prompt.ask("[bold yellow]Enter file extension (e.g. .m4v or press Enter for auto-detect)[/bold yellow]", default="")
                if custom_ext and not custom_ext.startswith("."):
                    custom_ext = f".{custom_ext}"
                return "CUSTOM", custom_ext.lower()
        except ValueError:
            pass
        console.print("[red]Invalid selection. Please enter a valid number.[/red]")


def get_file_path(expected_ext: str) -> Path:
    """Prompt for media file path and validate existence."""
    console.print("\n[bold cyan]Step 3: Enter File Path[/bold cyan]")
    console.print("[dim]Tip: You can drag and drop your file into this terminal, or type relative/absolute path.[/dim]")

    while True:
        raw_path = Prompt.ask("[bold green]File path[/bold green]")
        cleaned = raw_path.strip().strip("'\"")
        if not cleaned:
            console.print("[red]Path cannot be empty.[/red]")
            continue

        target = Path(cleaned).expanduser().resolve()

        if target.is_file():
            # Check extension warning if mismatch
            if expected_ext and target.suffix.lower() != expected_ext.lower():
                console.print(
                    f"[yellow]⚠️  Notice: File has extension '{target.suffix}', while '{expected_ext}' was selected.[/yellow]"
                )
                if not Confirm.ask("Do you want to proceed with this file?", default=True):
                    continue
            return target
        else:
            console.print(f"[red]❌ File not found at: {target}[/red]")
            # Look in current folder for convenience
            same_name_local = Path.cwd() / cleaned
            if same_name_local.is_file():
                return same_name_local.resolve()


def configure_transcription(file_path: Path, gpu_info: Dict[str, Any]) -> Dict[str, Any]:
    """Configure model, hardware, language, and output settings."""
    file_size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)
    rec_dev, rec_compute, rec_model = get_recommended_settings()

    console.print("\n[bold cyan]Step 4: Configuration & Settings[/bold cyan]")
    
    summary_table = Table(show_header=False, border_style="cyan")
    summary_table.add_column("Key", style="bold white", width=22)
    summary_table.add_column("Value", style="cyan")

    summary_table.add_row("Input File", str(file_path))
    summary_table.add_row("File Size", f"{file_size_mb} MB")
    summary_table.add_row("Target Hardware", f"{gpu_info.get('device_name', TARGET_GPU_NAME)} (RTX 3050)")
    summary_table.add_row("Compute Engine", f"{rec_dev.upper()} ({rec_compute})")
    summary_table.add_row("Whisper Model", f"{rec_model} (State-of-the-art accuracy)")
    console.print(summary_table)

    customize = Confirm.ask("\n[bold yellow]Do you want to customize advanced settings (Model, Language, VAD)?[/bold yellow]", default=False)
    
    selected_model = rec_model
    selected_lang = None
    vad_filter = DEFAULT_VAD_FILTER
    beam_size = DEFAULT_BEAM_SIZE

    if customize:
        console.print("\n[bold magenta]Available Whisper Models:[/bold magenta]")
        for key, m in WHISPER_MODELS.items():
            rec_tag = " [bold green](Recommended)[/bold green]" if m.get("recommended") else ""
            console.print(f"  • [bold cyan]{key}[/bold cyan]: {m['description']} - VRAM {m['vram']}{rec_tag}")
        selected_model = Prompt.ask("Choose model", choices=list(WHISPER_MODELS.keys()), default=rec_model)

        lang_input = Prompt.ask("Language code (e.g. 'en', 'hi', 'es' or press Enter for auto-detect)", default="")
        selected_lang = lang_input.strip() if lang_input.strip() else None

        vad_filter = Confirm.ask("Enable Voice Activity Filter (VAD) to filter silence/music?", default=False)
        beam_size = IntPrompt.ask("Beam search size (higher = better correctness)", default=DEFAULT_BEAM_SIZE)

    output_dir_str = Prompt.ask("Output directory", default=str(DEFAULT_OUTPUT_DIR))
    output_dir = Path(output_dir_str).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "file_path": file_path,
        "model_size": selected_model,
        "device": rec_dev,
        "compute_type": rec_compute,
        "language": selected_lang,
        "vad_filter": vad_filter,
        "beam_size": beam_size,
        "output_dir": output_dir,
    }


def run_pipeline() -> int:
    """Main execution orchestrator."""
    setup_cuda_environment()
    gpu_info = get_gpu_info()
    display_banner(gpu_info)

    # Step 1: Media Type
    media_type = select_media_type()

    # Step 2: Format
    fmt_name, ext = select_format(media_type)
    console.print(f"[green]✔ Selected Format: {fmt_name} ({ext or 'auto'})[/green]")

    # Step 3: File Path
    file_path = get_file_path(ext)
    console.print(f"[green]✔ File Selected: {file_path.name}[/green]")

    # Step 4: Configuration
    config = configure_transcription(file_path, gpu_info)

    # Confirmation
    if not Confirm.ask("\n[bold green]Ready to start transcription? Proceed?[/bold green]", default=True):
        console.print("[yellow]Operation cancelled by user.[/yellow]")
        return 0

    # Step 5: Loading Model
    console.print("\n" + "=" * 65)
    console.print(f"[bold cyan]⚡ Loading Whisper Model '{config['model_size']}' onto {config['device'].upper()} ({config['compute_type']})...[/bold cyan]")
    console.print("[dim]First run will automatically download model weights into cache.[/dim]")

    try:
        engine = TranscriptionEngine(
            model_size=config["model_size"],
            device=config["device"],
            compute_type=config["compute_type"],
        )
    except Exception as e:
        console.print(f"[bold red]❌ Failed to load model: {e}[/bold red]")
        return 1

    actual_dev_text = f"{engine.actual_device.upper()} ({engine.actual_compute_type})"
    console.print(f"[bold green]✔ Model loaded successfully on {actual_dev_text}![/bold green]\n")

    # Step 6: Processing with Real-Time Progress & Live Transcript Stream
    console.print("[bold cyan]🎙️  Transcribing media... Live preview below:[/bold cyan]\n")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[bold green]{task.percentage:>3.1f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    task_id = progress.add_task("Transcribing audio...", total=100.0)
    progress.start()

    def on_segment(seg: Dict[str, Any]) -> None:
        start_ts = format_timestamp(seg["start"])
        end_ts = format_timestamp(seg["end"])
        text = seg["text"].strip()
        progress.console.print(f"[dim cyan][{start_ts} -> {end_ts}][/dim cyan] [white]{text}[/white]")

    def on_progress(pct: float, current_sec: float, total_sec: float) -> None:
        progress.update(task_id, completed=pct)

    try:
        segments, info_dict = engine.transcribe(
            config["file_path"],
            language=config["language"],
            vad_filter=config["vad_filter"],
            beam_size=config["beam_size"],
            word_timestamps=DEFAULT_WORD_TIMESTAMPS,
            on_segment=on_segment,
            on_progress=on_progress,
        )
    except Exception as e:
        progress.stop()
        console.print(f"\n[bold red]❌ Transcription failed: {e}[/bold red]")
        return 1
    finally:
        progress.stop()

    console.print("\n[bold green]✔ Transcription completed successfully![/bold green]")

    # Step 7: Export Outputs
    out_dir: Path = config["output_dir"]
    base_name = file_path.stem

    txt_path = out_dir / f"{base_name}_transcript.txt"
    srt_path = out_dir / f"{base_name}_transcript.srt"
    vtt_path = out_dir / f"{base_name}_transcript.vtt"
    json_path = out_dir / f"{base_name}_transcript.json"
    words_path = out_dir / f"{base_name}_words.txt"
    md_path = out_dir / f"{base_name}_transcript.md"

    metadata = {
        "source_filepath": str(file_path),
        "source_filename": file_path.name,
        "model": config["model_size"],
        "device": engine.actual_device,
        "compute_type": engine.actual_compute_type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "beam_size": config["beam_size"],
        "vad_filter": config["vad_filter"],
    }

    export_txt(segments, txt_path, include_timestamps=True)
    export_srt(segments, srt_path)
    export_vtt(segments, vtt_path)
    export_words(segments, words_path)
    export_json(segments, info_dict, metadata, json_path)
    export_markdown(segments, info_dict, metadata, md_path)

    # Step 8: Results Summary Dashboard
    total_words = sum(len(s.get("words", [])) for s in segments)
    dur_str = format_timestamp(info_dict["duration"])
    lang_code = info_dict.get("language", "unknown").upper()
    lang_prob = info_dict.get("language_probability", 1.0) * 100

    results_table = Table(title="🎉 Transcription Results Summary", header_style="bold magenta", border_style="green")
    results_table.add_column("Metric", style="bold cyan", width=22)
    results_table.add_column("Details", style="white")

    results_table.add_row("Audio Duration", f"{dur_str} ({info_dict['duration']:.2f}s)")
    results_table.add_row("Detected Language", f"{lang_code} ({lang_prob:.1f}% confidence)")
    results_table.add_row("Processing Time", f"{info_dict.get('processing_time_seconds', 0.0):.2f} seconds")
    results_table.add_row("Speed Factor", f"{info_dict.get('speed_factor', 0.0):.1f}x real-time")
    results_table.add_row("Total Segments", str(len(segments)))
    results_table.add_row("Total Words", str(total_words))
    console.print(results_table)

    files_table = Table(title="📁 Generated Output Files", header_style="bold yellow", border_style="cyan")
    files_table.add_column("Format", style="bold white", width=12)
    files_table.add_column("File Path", style="cyan")

    files_table.add_row("Plain Text", str(txt_path))
    files_table.add_row("Subtitles SRT", str(srt_path))
    files_table.add_row("WebVTT", str(vtt_path))
    files_table.add_row("Word Timings", str(words_path))
    files_table.add_row("Structured JSON", str(json_path))
    files_table.add_row("Markdown", str(md_path))
    console.print(files_table)

    console.print(f"\n[bold green]✨ All output files have been saved to: [underline]{out_dir}[/underline][/bold green]\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run_pipeline())
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user. Exiting...[/yellow]")
        sys.exit(130)
