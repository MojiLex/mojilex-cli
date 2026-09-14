import shutil
import subprocess
from pathlib import Path

import pytest

from mojilex_cli.media import MediaLimits, worker


@pytest.mark.parametrize(
    ("timestamps", "durations", "wanted", "selected"),
    [
        (["0", "0.1", "0.2"], ["0.1"] * 3, 0.28, 2),
        (["0", "0.1", "0.2"], ["0.1"] * 3, 0.1, 1),
        (["0", "0.1", "0.2"], ["0.1"] * 3, 0.31, None),
        (["0", "0.15", "0.2"], ["0.1"] * 3, 0.28, 2),
        (["0", "0.05", "0.2"], ["0.1"] * 3, 0.28, 2),
        (["0", "0.1", "0.2"], ["0.1", None, "0.1"], 0.28, 2),
        (["0", "0.1", "0.2"], ["0.1", "0.1", "0.2"], 0.28, 2),
        (["0.01", "0.1", "0.2"], ["0.1"] * 3, 0.28, None),
        (["0", "0.2", "0.1"], ["0.1"] * 3, 0.28, None),
        (["0", "0.1", "0.3"], ["0.1"] * 3, 0.28, None),
        (["0", None, "0.2"], ["0.1"] * 3, 0.28, None),
    ],
)
def test_held_frame_requires_proven_presentation_timestamps(
    tmp_path, monkeypatch, timestamps, durations, wanted, selected
) -> None:
    metadata = [
        {"best_effort_timestamp_time": timestamp, "pkt_duration_time": duration}
        for timestamp, duration in zip(timestamps, durations, strict=True)
    ]
    monkeypatch.setattr(worker, "_webm_frame_metadata", lambda *a, **kw: metadata)
    kwargs = {"timestamp": wanted, "duration_ms": 300, "ffprobe": "unused", "timeout": 15}
    if selected is None:
        with pytest.raises(RuntimeError, match="cannot be proven"):
            worker._held_webm_frame_index(tmp_path / "media.webm", **kwargs)
    else:
        assert worker._held_webm_frame_index(tmp_path / "media.webm", **kwargs) == selected


def test_webm_durations_follow_presentation_timestamps_not_rounded_packet_durations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = [
        {"best_effort_timestamp_time": timestamp, "duration_time": "0.030000"}
        for timestamp in ("0", "0.030", "0.061", "0.091", "0.121")
    ]
    monkeypatch.setattr(worker, "_webm_frame_metadata", lambda *a, **kw: metadata)

    durations = worker._webm_frame_durations(
        tmp_path / "media.webm", ffprobe="unused", duration_ms=151, timeout=15
    )

    assert durations == (30_000, 31_000, 30_000, 30_000, 30_000)
    assert sum(durations) == 151_000


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg system prerequisite is not installed",
)
def test_three_frame_webm_holds_the_final_frame_at_last_midpoint(tmp_path: Path) -> None:
    source = tmp_path / "three.webm"
    frames = b"".join(
        bytes(color) * 256 for color in [(255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128)]
    )
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            "16x16",
            "-r",
            "10",
            "-i",
            "pipe:0",
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuva420p",
            "-lossless",
            "1",
            "-an",
            "-y",
            str(source),
        ],
        input=frames,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0
    decoded = worker._render_webm(
        source,
        tmp_path,
        300,
        MediaLimits(frames=8),
        "ffmpeg",
        codec="vp9",
        preserve_alpha=True,
    )
    try:
        assert len(decoded) == 8
        red, green, blue, alpha = decoded[-1].getpixel((0, 0))
        assert red < 10 and green < 10 and blue > 240
        assert 120 <= alpha <= 135
    finally:
        for frame in decoded:
            frame.close()
