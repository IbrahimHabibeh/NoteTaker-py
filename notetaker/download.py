"""Audio acquisition via yt-dlp (URLs) or direct local files."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://", "www."))


def download_audio(source: str, out_dir: Path) -> Tuple[Path, str]:
    """Resolve `source` to a local audio file and return (path, title).

    If `source` is a URL, download the best audio track via yt-dlp.
    If `source` is a local file, return it as-is.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if not is_url(source):
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return path, path.stem

    # Lazy import: yt-dlp is only needed for URL inputs.
    from yt_dlp import YoutubeDL

    outtmpl = str(out_dir / "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source, download=True)
        title = info.get("title") or info.get("id") or "audio"
        audio_path = out_dir / f"{info['id']}.mp3"

    if not audio_path.exists():
        raise RuntimeError(f"yt-dlp did not produce expected file: {audio_path}")
    return audio_path, title
