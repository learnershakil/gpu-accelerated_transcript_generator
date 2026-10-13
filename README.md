# 🎙️ GPU-Accelerated Transcript Generator

An interactive, high-accuracy audio and video transcription pipeline powered by **Faster-Whisper** and optimized for **NVIDIA GeForce RTX 3050** using **CUDA** (`float16`).

---

## ✨ Features

- **🎯 Interactive Guided CLI**:
  - **Step 1**: Choose media type (**Video** or **Audio**).
  - **Step 2**: Select from the **Top 10 Most Popular Formats** (or choose custom/auto-detect).
  - **Step 3**: Provide file path (drag-and-drop support, path expansion, automatic validation).
  - **Step 4**: Optional settings customization (Whisper model, forced language, VAD filter).
- **⚡ Hardware Accelerated**:
  - Automatically targets **NVIDIA GeForce RTX 3050 Laptop GPU**.
  - Uses **CUDA** with `float16` tensor execution for fast inference.
  - Graceful fallback to CPU `int8` if CUDA is unavailable.
- **🏆 High Correctness & Completeness**:
  - Defaults to `large-v3` for maximum multilingual accuracy.
  - Beam search (width = 5) and precise word-level timestamps.
  - Optional **Silero VAD** (Voice Activity Detection) to filter out background noise.
  - Auto-detects language and displays detection confidence.
- **📄 Multiple Output Formats Generated Automatically**:
  - `*_transcript.txt` : Clean readable text with timestamped segments.
  - `*_transcript.srt` : Standard SubRip subtitle file for video players (VLC, YouTube, Premiere).
  - `*_transcript.vtt` : WebVTT format for modern web video players.
  - `*_words.txt`      : Granular word-by-word timestamp log with confidence scores.
  - `*_transcript.json`: Full structured JSON containing all segments, words, and metadata.
  - `*_transcript.md`  : Beautifully formatted Markdown report with summary stats.

---

## 📂 Supported Formats

### 🎥 Top 10 Video Formats
1. **MP4** (`.mp4`) - Universal standard
2. **MKV** (`.mkv`) - Matroska Video
3. **MOV** (`.mov`) - Apple QuickTime Movie
4. **AVI** (`.avi`) - Audio Video Interleave
5. **WEBM** (`.webm`) - HTML5 Web Video
6. **FLV** (`.flv`) - Flash Video
7. **WMV** (`.wmv`) - Windows Media Video
8. **M4V** (`.m4v`) - Apple iTunes Video
9. **TS** (`.ts`) - MPEG Transport Stream
10. **3GP** (`.3gp`) - Mobile Multimedia
*(Plus custom / any format via option 11)*

### 🎵 Top 10 Audio Formats
1. **MP3** (`.mp3`) - MPEG Audio Layer III
2. **WAV** (`.wav`) - Uncompressed PCM Audio
3. **M4A** (`.m4a`) - AAC / ALAC Audio
4. **FLAC** (`.flac`) - Free Lossless Audio
5. **AAC** (`.aac`) - Advanced Audio Coding
6. **OGG** (`.ogg`) - Ogg Vorbis
7. **WMA** (`.wma`) - Windows Media Audio
8. **OPUS** (`.opus`) - Interactive Audio Codec
9. **AIFF** (`.aiff`) - Apple Interchange Audio
10. **AMR** (`.amr`) - Voice recordings
*(Plus custom / any format via option 11)*

---

## 🚀 Quick Start

### 1. Run Interactively
Launch the guided step-by-step pipeline:
```bash
./run.sh
```
or via the Python virtual environment:
```bash
.venv/bin/python main.py
```

### 2. Run Directly via CLI Argument
To transcribe a specific file without going through the menus:
```bash
./run.sh /path/to/media/file.mp4
# or
.venv/bin/python transcribe_audio.py /path/to/media/file.mp4
```

---

## 🛠️ Project Architecture

```
transcript-generator/
├── main.py              # Interactive Rich-based CLI pipeline
├── transcribe_audio.py  # Direct CLI and backward-compatible entry point
├── transcriber.py       # Core Faster-Whisper engine with streaming callbacks
├── gpu_utils.py         # NVIDIA GPU detection, CUDA 12 runtime linking
├── config.py            # Formats, models, and hardware settings
├── formatters.py        # Exporters for TXT, SRT, VTT, JSON, words, Markdown
├── requirements.txt     # Pinned Python package dependencies
├── run.sh               # One-click executable launcher
└── transcripts/         # Default directory where generated files are saved
```

---

## ⚙️ GPU & Performance Notes

- **Target GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (Ampere architecture, Compute Capability 8.6).
- **VRAM Footprint**:
  - `large-v3`: ~3.5 - 4.2 GB VRAM in `float16`.
  - `large-v3-turbo`: ~2.4 GB VRAM (faster alternative with comparable accuracy).
  - `medium`: ~1.5 GB VRAM.
- **Tip**: If you just performed a system driver update via pacman, rebooting once ensures the active kernel module matches the userspace NVIDIA libraries.
