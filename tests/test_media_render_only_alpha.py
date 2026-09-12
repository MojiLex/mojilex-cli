"""Real WebM resume regressions using generated, MIT-licensed color frames."""

import shutil
import subprocess
from pathlib import Path

import pytest

from mojilex_cli.media import MediaLimits
from mojilex_cli.media import worker as worker_module


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg system prerequisite is not installed",
)
@pytest.mark.parametrize("transparent_frames", [1, 20])
def test_render_only_alpha_webm_preserves_cached_render_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transparent_frames: int
) -> None:
    source = tmp_path / "alpha.webm"
    frames = b"".join(
        bytes((220, 30, 10, 128 if index < transparent_frames else 255)) * (16 * 16)
        for index in range(20)
    )
    encoded = subprocess.run(
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
            "20",
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
        timeout=20,
        check=False,
    )
    assert encoded.returncode == 0, encoded.stderr.decode(errors="replace")
    limits = MediaLimits(frames=4)
    initial = worker_module.process(
        source,
        tmp_path / "initial",
        "webm",
        limits,
        needs_repainting=False,
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        rlottie_renderer="unused",
    )
    assert initial["analysis"]["rendering"]["alpha_mode"] == "translucent"

    def fail_analysis(*args: object, **kwargs: object) -> object:
        pytest.fail("render-only resume must not recompute full-stream analysis")

    monkeypatch.setattr(worker_module, "_analyze_webm_full_stream", fail_analysis)
    resumed = worker_module.process(
        source,
        tmp_path / "resumed",
        "webm",
        limits,
        needs_repainting=False,
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        rlottie_renderer="unused",
        render_only=True,
        expected_dark_render=True,
    )
    assert resumed["analysis"] is None
    assert resumed["metadata"] == initial["metadata"]
    for key in ("frame_paths", "dark_frame_paths"):
        assert len(resumed[key]) == 4
        assert [Path(path).read_bytes() for path in resumed[key]] == [
            Path(path).read_bytes() for path in initial[key]
        ]

    # Even when the four sampled frames happen to be opaque, an alpha-capable
    # source must retain the verified full-stream background render context.
    with pytest.raises(RuntimeError, match="cached render context contradicts"):
        worker_module.process(
            source,
            tmp_path / "contradictory",
            "webm",
            limits,
            needs_repainting=False,
            ffmpeg="ffmpeg",
            ffprobe="ffprobe",
            rlottie_renderer="unused",
            render_only=True,
            expected_dark_render=False,
        )
