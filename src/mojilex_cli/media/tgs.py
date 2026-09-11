"""TGS processor facade."""

from pathlib import Path

from .models import ProcessedMedia
from .sandbox import SafeMediaWorker


def process_tgs(
    path: Path,
    output_dir: Path,
    *,
    worker: SafeMediaWorker | None = None,
    needs_repainting: bool = False,
) -> ProcessedMedia:
    return (worker or SafeMediaWorker()).process(
        path, output_dir, expected_format="tgs", needs_repainting=needs_repainting
    )
