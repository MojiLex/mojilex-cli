"""WebP processor facade."""

from pathlib import Path

from .models import ProcessedMedia
from .sandbox import SafeMediaWorker


def process_webp(
    path: Path, output_dir: Path, *, worker: SafeMediaWorker | None = None
) -> ProcessedMedia:
    return (worker or SafeMediaWorker()).process(path, output_dir, expected_format="webp")
