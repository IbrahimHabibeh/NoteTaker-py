"""Typer CLI orchestrating the full pipeline with rich progress output."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from . import chunk as chunk_mod
from . import download, export, hierarchy, nlp, summarize
from . import transcribe as transcribe_mod
from .models import Notes, TranscriptChunk

app = typer.Typer(add_completion=False, help="Generate hierarchical notes from audio or YouTube.")
console = Console()


@app.command()
def run(
    source: str = typer.Argument(..., help="YouTube URL or local audio/video file."),
    output: Path = typer.Option(Path("out"), "--output", "-o", help="Output directory."),
    model: str = typer.Option("base.en", "--model", "-m", help="faster-whisper model size."),
    embedder_name: str = typer.Option(
        "all-MiniLM-L6-v2", "--embedder", "-e", help="sentence-transformers model."
    ),
    chunk_seconds: float = typer.Option(60.0, "--chunk-seconds", help="Target chunk duration."),
    bullets: int = typer.Option(3, "--bullets", "-b", help="Maximum bullets per chunk."),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Force language code."),
    device: str = typer.Option("auto", "--device", help="Whisper device (auto/cpu/cuda)."),
    compute_type: str = typer.Option("auto", "--compute-type", help="Whisper compute type."),
    summarizer_model: str = typer.Option(
        nlp.DEFAULT_SUMMARIZER, "--summarizer-model", help="HuggingFace abstractive summarizer."
    ),
    abstractive: bool = typer.Option(
        True, "--abstractive/--no-abstractive", help="Rewrite summaries with a transformer."
    ),
    prefer_youtube_transcript: bool = typer.Option(
        True,
        "--prefer-youtube-transcript/--no-prefer-youtube-transcript",
        help="Try yt-dlp captions first for YouTube sources and skip Whisper when available.",
    ),
) -> None:
    """Run the full notetaker pipeline end-to-end."""
    output.mkdir(parents=True, exist_ok=True)
    audio_dir = output / "audio"

    console.print(Panel.fit(f"[bold]NoteTaker[/bold]\n[dim]{source}[/dim]"))

    segments: list[TranscriptChunk] = []
    title = "audio"

    if prefer_youtube_transcript and download.is_url(source):
        with console.status("[bold cyan]Trying YouTube transcript via yt-dlp..."):
            transcript_result = download.try_fetch_youtube_transcript(source, language=language)
        if transcript_result is not None:
            title = transcript_result.title
            segments = transcript_result.segments
            console.print(
                f"[green]✓[/green] Using yt-dlp {transcript_result.source_kind} "
                f"({len(segments)} segments)"
            )
        else:
            console.print("[yellow]•[/yellow] No YouTube transcript found, falling back to Whisper")

    if not segments:
        with console.status("[bold cyan]Fetching audio..."):
            audio_path, title = download.download_audio(source, audio_dir)
        console.print(f"[green]✓[/green] Audio ready: [dim]{audio_path.name}[/dim]")

        with console.status(f"[bold cyan]Transcribing with faster-whisper ({model})..."):
            segments = transcribe_mod.transcribe(
                audio_path, model_size=model, device=device, compute_type=compute_type, language=language
            )
        console.print(f"[green]✓[/green] Transcribed {len(segments)} segments")

    chunks = chunk_mod.merge_segments(segments, target_seconds=chunk_seconds)
    console.print(f"[green]✓[/green] Merged into {len(chunks)} timestamped chunks")

    with console.status("[bold cyan]Loading sentence-transformer..."):
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(embedder_name)

    summarizer_fn = None
    title_fn = None
    if abstractive:
        with console.status(f"[bold cyan]Loading abstractive model ({summarizer_model})..."):
            summarizer_fn = nlp.get_summarizer(summarizer_model)
            title_fn = nlp.get_headline_generator(summarizer_model)

    with console.status("[bold cyan]Summarizing chunks..."):
        chunk_summaries = summarize.summarize_chunks(
            chunks, embedder, bullets_per_chunk=bullets, summarizer=summarizer_fn
        )
    console.print(f"[green]✓[/green] Summarized {len(chunk_summaries)} chunks")

    with console.status("[bold cyan]Building hierarchical sections..."):
        sections = hierarchy.build_sections(chunk_summaries, embedder, title_fn=title_fn)
    console.print(f"[green]✓[/green] Organized into {len(sections)} topic sections")

    duration = chunks[-1].end if chunks else 0.0
    notes = Notes(title=title, source=source, duration=duration, sections=sections)

    md_path = output / f"{_slug(title)}.md"
    transcript_path = output / f"{_slug(title)}.transcript.json"
    notes_path = output / f"{_slug(title)}.notes.json"

    export.export_markdown(notes, md_path)
    export.export_transcript_json(segments, transcript_path)
    export.export_notes_json(notes, notes_path)

    console.print(
        Panel.fit(
            f"[bold green]Done.[/bold green]\n"
            f"Notes:      [cyan]{md_path}[/cyan]\n"
            f"Transcript: [cyan]{transcript_path}[/cyan]\n"
            f"Notes JSON: [cyan]{notes_path}[/cyan]"
        )
    )


def _slug(name: str) -> str:
    import re
    s = re.sub(r"[^\w\-]+", "-", name.strip().lower()).strip("-")
    return s or "notes"


if __name__ == "__main__":
    app()
