from __future__ import annotations

import mmap
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from mojilex_cli.media import MediaDependencyError, MediaLimits, SafeMediaWorker, unix_worker
from mojilex_cli.media import sandbox as sandbox_module


def test_unix_limits_preserve_all_caps_and_stricter_host_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, tuple[int, int]] = {}
    names = ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_CPU", "RLIMIT_FSIZE", "RLIMIT_NOFILE")
    resource = SimpleNamespace(
        **dict.fromkeys(names),
        RLIM_INFINITY=-1,
        getrlimit=lambda kind: (20, 25) if kind == "RLIMIT_CPU" else (-1, -1),
        setrlimit=lambda kind, limits: calls.update({kind: limits}),
    )
    for name in names:
        setattr(resource, name, name)
    monkeypatch.setitem(sys.modules, "resource", resource)
    memory = 256 * 1024 * 1024
    unix_worker.apply_limits(memory)
    assert calls == {
        "RLIMIT_AS": (memory, memory),
        "RLIMIT_DATA": (memory, memory),
        "RLIMIT_CPU": (20, 20),
        "RLIMIT_FSIZE": (64 * 1024 * 1024,) * 2,
        "RLIMIT_NOFILE": (64, 64),
    }


def test_unix_limit_failure_is_named_and_does_not_continue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def reject(kind: str, limits: tuple[int, int]) -> None:
        calls.append(kind)
        raise OSError("invalid argument")

    monkeypatch.setitem(
        sys.modules,
        "resource",
        SimpleNamespace(
            RLIMIT_AS="address-space",
            RLIM_INFINITY=-1,
            getrlimit=lambda kind: (-1, -1),
            setrlimit=reject,
        ),
    )
    with pytest.raises(RuntimeError, match="cannot apply worker limit RLIMIT_AS"):
        unix_worker.apply_limits(128 * 1024 * 1024)
    assert calls == ["address-space"]


@pytest.mark.parametrize("fails", [False, True])
def test_bootstrap_sets_limits_before_loading_decoder_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fails: bool
) -> None:
    events: list[str] = []

    def apply(memory: int) -> None:
        assert memory == 134217728
        events.append("limits")
        if fails:
            raise RuntimeError("cannot apply worker limit RLIMIT_AS")

    def run(module: str, *, run_name: str) -> None:
        assert module == "mojilex_cli.media.worker" and run_name == "__main__"
        assert sys.argv == [module, "--source", "fixture.webp"]
        events.append("decoder")

    monkeypatch.setattr(unix_worker, "apply_limits", apply)
    monkeypatch.setitem(sys.modules, "runpy", SimpleNamespace(run_module=run))
    monkeypatch.setattr(sys, "argv", ["bootstrap", "134217728", "--source", "fixture.webp"])
    if fails:
        with pytest.raises(SystemExit) as captured:
            unix_worker.main()
        assert captured.value.code == 1
        assert events == ["limits"]
        assert "RLIMIT_AS" in capsys.readouterr().err
    else:
        unix_worker.main()
        assert events == ["limits", "decoder"]


def test_unix_worker_launch_uses_fresh_bootstrap_without_preexec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "fixture.webp"
    Image.new("RGB", (2, 2), "red").save(source, "WEBP")
    monkeypatch.setattr(sandbox_module, "os", SimpleNamespace(name="posix", environ=os.environ))

    def launch(command: list[str], **kwargs: object) -> None:
        assert command[1] == "-P"
        assert Path(command[2]).name == "unix_worker.py"
        assert command[3] == str(MediaLimits().worker_memory_bytes)
        assert command[4:6] == ["--source", str(source.resolve())]
        assert "preexec_fn" not in kwargs
        assert kwargs["start_new_session"] is True
        raise OSError("synthetic launch stop")

    monkeypatch.setattr(sandbox_module.subprocess, "Popen", launch)
    with pytest.raises(MediaDependencyError, match="cannot start"):
        SafeMediaWorker().process(source, tmp_path / "output", expected_format="webp")


def test_direct_unix_bootstrap_does_not_shadow_stdlib_inspect(tmp_path: Path) -> None:
    # This launches the actual entrypoint even on Windows. Only resource is a
    # test substitute: no untrusted image is decoded, and all five configured
    # limits must be requested before the real decoder dependency stack imports.
    (tmp_path / "resource.py").write_text(
        "import sys\n"
        "RLIMIT_AS='as'; RLIMIT_DATA='data'; RLIMIT_CPU='cpu'\n"
        "RLIMIT_FSIZE='files'; RLIMIT_NOFILE='descriptors'; RLIM_INFINITY=-1\n"
        "expected={'as':536870912,'data':536870912,'cpu':30,'files':67108864,'descriptors':64}\n"
        "def getrlimit(kind): return (-1,-1)\n"
        "def setrlimit(kind,limits):\n"
        " assert not any(name.startswith(('PIL','mojilex_cli')) for name in sys.modules)\n"
        " assert limits == (expected.pop(kind),)*2\n"
        " if not expected: print('ALL_LIMITS_APPLIED')\n",
        encoding="utf-8",
    )
    environment = sandbox_module._worker_environment(tmp_path)
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path), environment["PYTHONPATH"]))
    result = subprocess.run(
        [sys.executable, "-P", str(Path(unix_worker.__file__)), "536870912"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    # Reaching the actual worker argument validator proves its imports worked;
    # deliberately omit image arguments so this portability test decodes nothing.
    assert result.returncode == 2, result.stderr
    assert "ALL_LIMITS_APPLIED" in result.stdout
    assert (
        "the following arguments are required: --source, --output, --format, --limits"
        in result.stderr
    )
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(os.name == "nt", reason="Unix process resource limit regression")
def test_unix_worker_starts_from_thread_with_parent_larger_than_worker_limit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fixture.webp"
    Image.new("RGB", (2, 2), "red").save(source, "WEBP")
    # Reserve address space without allocating physical pages. Darwin rejects an
    # RLIMIT_AS below this inherited mapping if applied between fork and exec.
    with (
        mmap.mmap(-1, 768 * 1024 * 1024) as reservation,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        assert len(reservation) > MediaLimits().worker_memory_bytes
        result = executor.submit(
            SafeMediaWorker().process, source, tmp_path / "output", expected_format="webp"
        ).result(timeout=40)
    assert result.metadata.width == 2
    assert len(result.frame_paths) == 1
