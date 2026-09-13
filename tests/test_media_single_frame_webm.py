import subprocess
from pathlib import Path

import pytest
from PIL import Image

from mojilex_cli.media import worker
from mojilex_cli.media.models import MediaLimits


@pytest.mark.parametrize("frame_count", [1, 2, 20])
def test_only_proven_single_frame_can_repeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frame_count: int,
) -> None:
    calls = []
    probes = []

    def decode(source, rendered, timestamp, limits, executable, **kwargs):
        calls.append(timestamp)
        assert kwargs == {"codec": "vp9", "preserve_alpha": True}
        if timestamp == 0:
            Image.new("RGBA", (100, 100), (10, 20, 30, 100)).save(rendered)
        return subprocess.CompletedProcess([], 0, b"", b"")

    def durations(source, **kwargs):
        probes.append(kwargs)
        return (40000,) * frame_count

    monkeypatch.setattr(worker.shutil, "which", lambda value: value)
    monkeypatch.setattr(worker, "_decode_webm_frame", decode)
    monkeypatch.setattr(worker, "_webm_frame_durations", durations)
    kwargs = dict(codec="vp9", preserve_alpha=True, ffprobe="selected-probe")
    if frame_count == 1:
        frames = worker._render_webm(
            tmp_path / "input.webm", tmp_path, 40, MediaLimits(), "ffmpeg", **kwargs
        )
        try:
            assert len(frames) == 8
            assert len({id(frame) for frame in frames}) == 8
            assert all(frame.getpixel((0, 0)) == (10, 20, 30, 100) for frame in frames)
            assert calls[-1] == 0
        finally:
            for frame in frames:
                frame.close()
    else:
        with pytest.raises(RuntimeError, match="deterministic WebM frame"):
            worker._render_webm(
                tmp_path / "input.webm", tmp_path, 40, MediaLimits(), "ffmpeg", **kwargs
            )
        assert 0 not in calls
    assert len(probes) == 1
    assert probes[0]["ffprobe"] == "selected-probe"


def test_failed_decode_does_not_trigger_single_frame_reinterpretation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker.shutil, "which", lambda value: value)
    monkeypatch.setattr(
        worker,
        "_decode_webm_frame",
        lambda *a, **kw: subprocess.CompletedProcess([], 1, b"", b"failure"),
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("decoder error must not reinterpret timeline")

    monkeypatch.setattr(worker, "_webm_frame_durations", unexpected)
    with pytest.raises(RuntimeError, match="deterministic WebM frame"):
        worker._render_webm(
            tmp_path / "input.webm",
            tmp_path,
            40,
            MediaLimits(),
            "ffmpeg",
            codec="vp9",
            preserve_alpha=True,
        )
