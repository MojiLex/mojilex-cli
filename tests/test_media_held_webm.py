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
        (["0", "0.1", "0.3"], ["0.1"] * 3, 0.28, 1),
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


@pytest.mark.parametrize("declared_ms", [1, 66])
@pytest.mark.parametrize("duration_key", ["duration_time", "pkt_duration_time"])
def test_short_container_duration_uses_final_decoded_frame(declared_ms, duration_key):
    frames = [
        {"best_effort_timestamp_time": t, duration_key: "0.033"} for t in ("0", "0.033", "0.066")
    ]
    intervals = worker._webm_presentation_intervals(frames, duration_ms=declared_ms)
    assert intervals == ((0, 33000), (33000, 66000), (66000, 99000))


@pytest.mark.parametrize("duration", [None, "0", "-1", "NaN", "Infinity"])
def test_short_container_requires_explicit_positive_final_duration(duration):
    frames = [
        {"best_effort_timestamp_time": "0"},
        {"best_effort_timestamp_time": "0.033", "duration_time": duration},
    ]
    with pytest.raises(worker.AnalysisError, match="no usable duration"):
        worker._webm_presentation_intervals(frames, duration_ms=1)


def test_recovered_endpoint_cannot_bypass_duration_limit():
    frames = [
        {"best_effort_timestamp_time": "0"},
        {"best_effort_timestamp_time": "9.99", "duration_time": "0.033"},
    ]
    with pytest.raises(worker.AnalysisError, match="duration limit"):
        worker._webm_presentation_intervals(frames, duration_ms=1)


@pytest.mark.parametrize("limit_ms", [100, 50])
def test_recovered_duration_is_used_before_ai_sampling(tmp_path, monkeypatch, limit_ms):
    from PIL import Image

    source = tmp_path / "source.webm"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(worker, "sniff_format", lambda _: "webm")
    monkeypatch.setattr(
        worker,
        "inspect_webm",
        lambda *a, **kw: {
            "width": 16,
            "height": 16,
            "duration_ms": 1,
            "codec": "vp9",
            "has_alpha": False,
        },
    )
    monkeypatch.setattr(worker, "_webm_frame_durations", lambda *a, **kw: (33000,) * 3)
    observed = []

    def render(source, output, duration, *a, **kw):
        observed.append(duration)
        return [Image.new("RGBA", (16, 16), (255, 0, 0, 255))]

    monkeypatch.setattr(worker, "_render_webm", render)
    kwargs = dict(
        needs_repainting=False,
        ffmpeg="unused",
        ffprobe="unused",
        rlottie_renderer="unused",
        render_only=True,
        expected_dark_render=True,
    )
    if limit_ms == 50:
        with pytest.raises(worker.MediaLimitError):
            worker.process(
                source, tmp_path / "out", "webm", MediaLimits(max_duration_ms=limit_ms), **kwargs
            )
        assert not observed
    else:
        result = worker.process(
            source, tmp_path / "out", "webm", MediaLimits(max_duration_ms=limit_ms), **kwargs
        )
        assert observed == [99]
        assert result["metadata"]["duration_ms"] == 99
