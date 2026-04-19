# Vid2SRT

A simple command-line tool that transcribes video and audio files into **SRT subtitle files** using [OpenAI Whisper](https://github.com/openai/whisper).

Works on a **single file** or an **entire folder** (optionally recursive), with a clean progress bar for both the current transcription and the overall batch.

---

## Features

- 🎬 Transcribes video *and* audio files (`.mp4`, `.mkv`, `.mov`, `.mp3`, `.wav`, and more)
- 📁 Accepts **either a single file or a folder** as input
- 🔁 Optional **recursive** folder scanning
- ⏭️ `--skip-existing` flag to skip files that already have an SRT next to them
- 📊 Two-level progress display — per-file tqdm bar inside an overall batch bar
- 🌍 Auto language detection, or pass a language code with `-l`
- 🖥️ GPU (CUDA) auto-detected; can be forced with `--device cuda|cpu`
- ✅ Clean processing summary at the end (successful / skipped / failed)

---

## Requirements

- **Python 3.9+**
- **[FFmpeg](https://ffmpeg.org/)** installed and available on your `PATH` (required by Whisper)
- Python packages from `requirements.txt`:
  - `openai-whisper`
  - `tqdm`

> Whisper depends on PyTorch, which will be installed automatically. If you want CUDA acceleration, install a CUDA-enabled build of PyTorch *before* installing the other requirements — see the [PyTorch install page](https://pytorch.org/get-started/locally/).

### Installing FFmpeg

| OS | Command |
|---|---|
| Ubuntu / Debian | `sudo apt update && sudo apt install ffmpeg` |
| macOS (Homebrew) | `brew install ffmpeg` |
| Windows (Chocolatey) | `choco install ffmpeg` |
| Windows (Scoop) | `scoop install ffmpeg` |

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/Vid2SRT.git
cd Vid2SRT

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate         # on Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Transcribe a single file

```bash
python vid2srt.py -i path/to/video.mp4
```

The SRT will be written next to the source file: `path/to/video.srt`.

### Transcribe a whole folder

```bash
python vid2srt.py -i ./videos -o ./subtitles
```

### Recursively process a folder tree

```bash
python vid2srt.py -i ./videos -r
```

### Specify language and a larger model

```bash
python vid2srt.py -i ./videos -l en -m medium -o ./subtitles
```

### Skip files that already have an SRT

```bash
python vid2srt.py -i ./videos --skip-existing
```

### Force CPU (no GPU)

```bash
python vid2srt.py -i video.mp4 --device cpu
```

---

## CLI Options

| Option | Description |
|---|---|
| `-i, --input` *(required)* | Path to a media file **or** a folder of media files. |
| `-o, --output` | Output directory for SRT files. Default: next to each source file. |
| `-m, --model` | Whisper model: `tiny`, `base`, `small`, `medium`, `large`, `large-v1/v2/v3`, `turbo`. Default: `base`. |
| `-l, --language` | Language code (e.g. `en`, `es`, `fr`, `fa`). Auto-detected if omitted. |
| `-r, --recursive` | Recursively scan subdirectories (folder input only). |
| `--skip-existing` | Skip files that already have a matching `.srt` file. |
| `--device` | Force device: `cuda` or `cpu`. Default: auto. |
| `--no-progress` | Disable the per-transcription progress bar (prints each segment instead). |
| `-v, --verbose` | Enable verbose logging. |
| `-h, --help` | Show full help message. |

---

## Supported Formats

**Video:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`, `.webm`
**Audio:** `.mp3`, `.wav`, `.m4a`, `.flac`, `.aac`, `.ogg`

---

## Choosing a Whisper Model

| Model | Parameters | VRAM (approx.) | Relative speed | Accuracy |
|---|---|---|---|---|
| `tiny` | 39 M | ~1 GB | ~32× | Lowest |
| `base` | 74 M | ~1 GB | ~16× | Low |
| `small` | 244 M | ~2 GB | ~6× | Medium |
| `medium` | 769 M | ~5 GB | ~2× | High |
| `large` / `large-v3` | 1550 M | ~10 GB | 1× | Highest |
| `turbo` | ~800 M | ~6 GB | ~8× | High (optimized) |

> On CPU, stick to `tiny`, `base`, or `small` — larger models will be very slow.

---

## Example Output

```
[*] Loading Whisper model: medium on cuda ...
[+] Model loaded in 4.2s

[*] Found 3 file(s) to process.

→ Transcribing: Lecture 1.mp4
  Transcribing: 100%|████████████████████████| 312000/312000 [03:12<00:00]
✓ Saved:   ./subtitles/Lecture 1.srt

→ Transcribing: Lecture 2.mp4
  Transcribing: 100%|████████████████████████| 298400/298400 [03:05<00:00]
✓ Saved:   ./subtitles/Lecture 2.srt

...

============================================================
                   PROCESSING SUMMARY
============================================================
Total files:          3
✓ Successful:         3
⊘ Skipped:            0
✗ Failed:             0
============================================================
```

---

## Troubleshooting

- **`FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`** — FFmpeg is not installed or not on your `PATH`. See the install table above.
- **CUDA out of memory** — Use a smaller model (`-m small` or `-m base`) or force CPU with `--device cpu`.
- **Transcription is very slow on CPU** — This is expected for `medium`/`large`. Use `tiny`/`base`, or run on a GPU machine.
- **Weird characters in SRT** — Make sure you open the file in a UTF-8 capable editor.

---

## License

MIT — see `LICENSE` file (feel free to add one when you create the repo).

---

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) for the speech recognition model
- [tqdm](https://github.com/tqdm/tqdm) for the progress bars
