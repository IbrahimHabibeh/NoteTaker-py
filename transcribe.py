"""Backwards-compatible entry point. Prefer `python -m notetaker`."""

from notetaker.cli import app

if __name__ == "__main__":
    app()