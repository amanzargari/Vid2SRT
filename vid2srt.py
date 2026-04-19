#!/usr/bin/env python3
"""
Vid2SRT - Transcribe video/audio files to SRT subtitles using OpenAI Whisper.

Supports a single file or a whole folder (optionally recursive), with a clean
progress display for both the current transcription and the overall batch.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import whisper
from whisper.utils import get_writer
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {
    # Video
    ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm",
    # Audio
    ".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg",
}

AVAILABLE_MODELS = [
    "tiny", "base", "small", "medium", "large",
    "large-v1", "large-v2", "large-v3", "turbo",
]


# --------------------------------------------------------------------------- #
# Progress-bar helpers                                                         #
# --------------------------------------------------------------------------- #
class _ColoredBar(tqdm):
    """A tqdm subclass with a cleaner, colored default style."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "bar_format",
            "  {desc}: {percentage:3.0f}%|{bar}| "
            "{n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
        kwargs.setdefault("ncols", 100)
        kwargs.setdefault("colour", "green")
        super().__init__(*args, **kwargs)


class _WhisperTqdmPatch:
    """
    Context manager that monkey-patches whisper.transcribe.tqdm with a nicer
    progress bar. Falls back silently if whisper's internal API changes.
    """

    def __init__(self, enable: bool = True):
        self.enable = enable
        self._patched = False
        self._original = None
        self._module = None

    def __enter__(self):
        if not self.enable:
            return self
        try:
            import whisper.transcribe as wt  # noqa: WPS433
            self._module = wt
            self._original = wt.tqdm
            wt.tqdm = _ColoredBar
            self._patched = True
        except Exception:  # pragma: no cover - defensive
            self._patched = False
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._patched and self._module is not None:
            try:
                self._module.tqdm = self._original
            except Exception:  # pragma: no cover - defensive
                pass
        return False


# --------------------------------------------------------------------------- #
# Core transcriber                                                             #
# --------------------------------------------------------------------------- #
class WhisperTranscriber:
    """Transcribe media files to SRT using OpenAI Whisper."""

    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        print(f"[*] Loading Whisper model: {model_size}"
              f"{' on ' + device if device else ''} ...")
        t0 = time.time()
        self.model = whisper.load_model(model_size, device=device)
        print(f"[+] Model loaded in {time.time() - t0:.1f}s\n")

    def transcribe_file(
        self,
        media_path: Path,
        output_dir: Path,
        language: Optional[str] = None,
        show_progress: bool = True,
    ) -> Path:
        """Transcribe one file and write its SRT next to it (or to output_dir)."""
        if not media_path.exists():
            raise FileNotFoundError(f"File not found: {media_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        options = {
            # verbose=False => whisper shows its built-in tqdm progress bar
            # verbose=True  => whisper prints every segment instead
            "verbose": False if show_progress else True,
        }
        if language:
            options["language"] = language

        with _WhisperTqdmPatch(enable=show_progress):
            result = self.model.transcribe(str(media_path), **options)

        # Write the SRT file. Whisper's writer derives the filename from the
        # original media path's stem, so we just pass the real path.
        srt_writer = get_writer("srt", str(output_dir))
        srt_writer(
            result,
            str(media_path),
            {"max_line_width": None, "max_line_count": None, "highlight_words": False},
        )

        return output_dir / f"{media_path.stem}.srt"

    @staticmethod
    def get_media_files(folder: Path, recursive: bool = False) -> List[Path]:
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder}")

        walker = folder.rglob if recursive else folder.glob
        files: List[Path] = []
        for ext in SUPPORTED_EXTENSIONS:
            # case-insensitive glob
            files.extend(walker(f"*{ext}"))
            files.extend(walker(f"*{ext.upper()}"))
        # dedupe & sort
        return sorted(set(files))

    def process_folder(
        self,
        folder: Path,
        output_dir: Optional[Path],
        language: Optional[str],
        recursive: bool,
        skip_existing: bool,
        show_progress: bool,
    ) -> dict:
        files = self.get_media_files(folder, recursive)
        total = len(files)

        results = {
            "total": total,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "files": [],
        }

        if total == 0:
            print(f"[!] No supported media files found in {folder}")
            return results

        print(f"[*] Found {total} file(s) to process.\n")

        batch_bar = tqdm(
            files,
            desc="Overall",
            unit="file",
            colour="blue",
            ncols=100,
            bar_format="{desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]",
        )

        for media_path in batch_bar:
            batch_bar.set_description(f"Overall ({media_path.name[:35]})")
            batch_bar.write(f"\n→ Transcribing: {media_path.name}")

            srt_dir = output_dir if output_dir else media_path.parent
            srt_path = srt_dir / f"{media_path.stem}.srt"

            if skip_existing and srt_path.exists():
                batch_bar.write(f"⊘ Skipped (SRT exists): {srt_path.name}")
                results["skipped"] += 1
                results["files"].append(
                    {"file": str(media_path), "status": "skipped",
                     "reason": "SRT already exists"}
                )
                continue

            try:
                out = self.transcribe_file(
                    media_path, srt_dir, language=language,
                    show_progress=show_progress,
                )
                batch_bar.write(f"✓ Saved:   {out}")
                results["processed"] += 1
                results["files"].append(
                    {"file": str(media_path), "status": "success",
                     "srt_path": str(out)}
                )
            except Exception as exc:  # noqa: BLE001
                batch_bar.write(f"✗ Failed:  {media_path.name} — {exc}")
                results["failed"] += 1
                results["files"].append(
                    {"file": str(media_path), "status": "failed",
                     "error": str(exc)}
                )

        batch_bar.close()
        return results


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def print_summary(results: dict) -> None:
    bar = "=" * 60
    print("\n" + bar)
    print("PROCESSING SUMMARY".center(60))
    print(bar)
    print(f"Total files:          {results['total']}")
    print(f"✓ Successful:         {results['processed']}")
    print(f"⊘ Skipped:            {results['skipped']}")
    print(f"✗ Failed:             {results['failed']}")
    print(bar)

    if results["failed"] > 0:
        print("\nFailed files:")
        for info in results["files"]:
            if info["status"] == "failed":
                print(f"  - {info['file']}: {info.get('error', 'unknown error')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vid2srt",
        description="Transcribe video/audio files to SRT subtitles using OpenAI Whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Single file, default (base) model, auto-detect language
  python vid2srt.py -i video.mp4

  # Entire folder (recursive), English, medium model, custom output dir
  python vid2srt.py -i ./videos -r -l en -m medium -o ./subtitles

  # Skip files that already have an SRT next to them
  python vid2srt.py -i ./videos --skip-existing

  # Force CPU and show whisper's per-segment logs instead of a progress bar
  python vid2srt.py -i video.mp4 --device cpu --no-progress
""",
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to a media file OR a folder containing media files.",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output directory for SRT files. "
             "Default: write each SRT next to its source file.",
    )
    parser.add_argument(
        "-m", "--model", default="base", choices=AVAILABLE_MODELS,
        help="Whisper model size (default: base). Larger = more accurate, slower.",
    )
    parser.add_argument(
        "-l", "--language", default=None,
        help="Language code (e.g. en, es, fr, fa). Auto-detect if omitted.",
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="Recursively process subdirectories (folder input only).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip files that already have a matching .srt file.",
    )
    parser.add_argument(
        "--device", default=None, choices=["cuda", "cpu"],
        help="Force device (default: auto-detect — uses CUDA if available).",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable the per-transcription progress bar "
             "(whisper will print each segment instead).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    input_path = Path(args.input).expanduser().resolve()
    output_dir = (
        Path(args.output).expanduser().resolve() if args.output else None
    )

    if not input_path.exists():
        print(f"ERROR: input path does not exist: {input_path}", file=sys.stderr)
        return 2

    try:
        transcriber = WhisperTranscriber(model_size=args.model, device=args.device)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not load Whisper model: {exc}", file=sys.stderr)
        return 3

    show_progress = not args.no_progress

    # --- single-file mode ----------------------------------------------------
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(
                f"ERROR: unsupported file extension: {input_path.suffix}\n"
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
                file=sys.stderr,
            )
            return 2

        srt_dir = output_dir if output_dir else input_path.parent
        print(f"→ Transcribing: {input_path.name}")
        try:
            srt_path = transcriber.transcribe_file(
                input_path, srt_dir,
                language=args.language, show_progress=show_progress,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"\n✗ Transcription failed: {exc}", file=sys.stderr)
            return 1

        print(f"\n✓ SRT file saved to: {srt_path}")
        return 0

    # --- folder mode ---------------------------------------------------------
    results = transcriber.process_folder(
        folder=input_path,
        output_dir=output_dir,
        language=args.language,
        recursive=args.recursive,
        skip_existing=args.skip_existing,
        show_progress=show_progress,
    )
    print_summary(results)
    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
