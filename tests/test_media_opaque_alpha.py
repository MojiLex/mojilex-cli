import io
from pathlib import Path

import pytest

from mojilex_cli.media import MediaLimits, worker


@pytest.mark.parametrize(
    ("payload", "returncode", "accepted"),
    [
        (b"\xff" * 8, 0, True),
        (b"\xff" * 7 + b"\x80", 0, False),
        (b"\xff" * 4, 0, False),
        (b"\xff" * 12, 0, False),
        (b"", 1, False),
        (b"\xff" * 8, 1, False),
    ],
)
def test_native_alpha_proof_requires_complete_opaque_plane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    returncode: int,
    accepted: bool,
) -> None:
    class Decoder:
        stdout = io.BytesIO(payload)

        def wait(self, **kwargs):
            return returncode

        def poll(self):
            return returncode

        def communicate(self, **kwargs):
            return b"", b""

    def popen(command, **kwargs):
        assert "-vsync" not in command
        assert command[command.index("-fps_mode") + 1] == "passthrough"
        assert command[command.index("-vf") + 1] == "alphaextract"
        assert "format=rgba" not in command
        assert command[command.index("-c:v") + 1] == "libvpx-vp9"
        return Decoder()

    monkeypatch.setattr(worker.subprocess, "Popen", popen)
    monkeypatch.setattr(worker, "_webm_frame_durations", lambda *a, **kw: (50000, 50000))
    info = {"width": 2, "height": 2, "codec": "vp9", "duration_ms": 100}
    if accepted:
        worker._verify_opaque_webm_alpha(
            tmp_path / "input.webm", info, MediaLimits(), ffmpeg="ffmpeg", ffprobe="ffprobe"
        )
    else:
        with pytest.raises(RuntimeError, match="alpha"):
            worker._verify_opaque_webm_alpha(
                tmp_path / "input.webm", info, MediaLimits(), ffmpeg="ffmpeg", ffprobe="ffprobe"
            )
