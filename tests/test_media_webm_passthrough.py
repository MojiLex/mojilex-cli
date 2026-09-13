"""Full-stream decoding uses modern FFmpeg passthrough without relaxing validation."""

import io
from pathlib import Path

import pytest

from mojilex_cli.analysis import AnalysisError
from mojilex_cli.media import MediaLimits, worker


@pytest.mark.parametrize(
    "frames,returncode,accepted",
    [(2, 0, True), (1, 0, False), (3, 0, False), (2, 1, False)],
)
def test_full_webm_passthrough_still_requires_exact_complete_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frames: int, returncode: int, accepted: bool
) -> None:
    class Decoder:
        stdout = io.BytesIO(bytes((255, 0, 0, 128)) * 4 * frames)

        def wait(self, **kwargs):
            return returncode

        def poll(self):
            return returncode

        def communicate(self, **kwargs):
            return b"", b""

    def popen(command, **kwargs):
        assert "-vsync" not in command  # Removed by FFmpeg 9.
        assert command[command.index("-fps_mode") + 1] == "passthrough"
        assert command[command.index("-c:v") + 1] == "libvpx-vp9"
        assert command[command.index("-pix_fmt") + 1] == "rgba"
        assert command.index("-fps_mode") > command.index("-i")
        return Decoder()

    monkeypatch.setattr(worker.subprocess, "Popen", popen)
    monkeypatch.setattr(worker, "_webm_frame_durations", lambda *a, **kw: (40000, 40000))
    monkeypatch.setattr(worker, "decoder_backend_fingerprint", lambda *a, **kw: "a" * 64)
    info = {"width": 2, "height": 2, "codec": "vp9", "duration_ms": 80, "has_alpha": True}
    if accepted:
        analysis = worker._analyze_webm_full_stream(
            tmp_path / "input.webm",
            info,
            MediaLimits(),
            ffmpeg="ffmpeg",
            ffprobe="ffprobe",
            needs_repainting=False,
        )
        assert analysis.rendering.alpha_mode == "translucent"
    else:
        with pytest.raises(AnalysisError, match="ffmpeg"):
            worker._analyze_webm_full_stream(
                tmp_path / "input.webm",
                info,
                MediaLimits(),
                ffmpeg="ffmpeg",
                ffprobe="ffprobe",
                needs_repainting=False,
            )
