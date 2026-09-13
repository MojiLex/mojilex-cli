from __future__ import annotations

import base64
import hashlib
import io
import json
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from mojilex_cli.commands import gallery, packs
from mojilex_cli.commands.runtime import CommandResult
from mojilex_cli.config import MojiLexConfig
from mojilex_cli.runs import ElementCheckpoint
from test_dataset_helpers import NATIVE_EMOJI_ID, write_fixture
from test_pack_commands import _save, _state


def _saved(text: str = "Мультяшный взрыв") -> CommandResult:
    return CommandResult(
        result={
            "pack": {"names": ["NewsEmoji"], "items": 1},
            "counts": {"ready": 1},
            "items": [
                {
                    "native_id": "123",
                    "descriptions": {
                        "ru": {"text": text, "motion": "Вспышка", "usage": ["Удивление"]},
                        "en": {"text": "Cartoon explosion"},
                    },
                    "content": {"rating": "general", "warnings": ["flashing"]},
                    "semantic_tags": ["explosion"],
                    "facets": {"styles": ["cartoon"]},
                }
            ],
        }
    )


def _frame(root: Path, *, descriptor: str = "a" * 64) -> bytes:
    entry = root / descriptor
    entry.mkdir(parents=True)
    stream = io.BytesIO()
    Image.new("RGBA", (256, 256), (255, 0, 0, 255)).save(stream, format="PNG")
    data = stream.getvalue()
    (entry / "light-00.png").write_bytes(data)
    (entry / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "fingerprint": "b" * 64,
                "frames": [
                    {
                        "name": "light-00.png",
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return data


def test_gallery_keeps_labels_and_details_and_escapes_all_untrusted_text() -> None:
    attack = '</script><script>alert("x")</script><img src=x onerror=alert(1)>'
    result = _saved(attack)
    result.result["pack"]["names"] = [attack]
    result.result["items"][0]["native_id"] = attack
    result.result["items"][0]["content"]["warnings"].append(attack)
    result.result["items"][0]["semantic_tags"].append(attack)
    result.warnings.append(attack)
    page = gallery.render_gallery(result, {})
    assert page.count("<script>") == 1
    assert attack not in page
    assert "&lt;script&gt;" in page
    assert "Мигающие вспышки" in page and "(flashing)" in page
    assert "Ручное одобрение не требуется" in page
    assert "Локальное изображение не сохранено" in page
    assert "<details><summary>English description</summary>" in page
    assert "<details><summary>Теги и подробности</summary>" in page
    assert "innerHTML" not in page
    assert "card.textContent" in page
    assert "https://" not in page and "http://" not in page
    digest = base64.b64encode(hashlib.sha256(gallery._SCRIPT.encode()).digest()).decode()
    assert f"sha256-{digest}" in page


def test_gallery_reads_verified_frame_without_modifying_it(tmp_path: Path) -> None:
    data = _frame(tmp_path)
    before = _state(tmp_path)
    assert gallery._preview(tmp_path, "a" * 64) == data
    page = gallery.render_gallery(_saved(), {"123": data})
    assert "data:image/png;base64," + base64.b64encode(data).decode() in page
    assert _state(tmp_path) == before


@pytest.mark.parametrize("corruption", ["checksum", "path", "size", "format", "missing"])
def test_bad_preview_is_a_fallback(tmp_path: Path, corruption: str) -> None:
    _frame(tmp_path)
    entry = tmp_path / ("a" * 64)
    manifest = json.loads((entry / "manifest.json").read_text())
    frame = manifest["frames"][0]
    if corruption == "checksum":
        frame["sha256"] = "f" * 64
    elif corruption == "path":
        frame["name"] = "../../private.png"
    elif corruption == "size":
        frame["bytes"] = gallery._MAX_PREVIEW_BYTES + 1
    elif corruption == "format":
        data = b"<svg onload='alert(1)'></svg>"
        (entry / "light-00.png").write_bytes(data)
        frame.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    else:
        (entry / "light-00.png").unlink()
    (entry / "manifest.json").write_text(json.dumps(manifest))
    before = _state(tmp_path)
    assert gallery._preview(tmp_path, "a" * 64) is None
    assert gallery._preview(tmp_path, "../../outside") is None
    assert _state(tmp_path) == before


def test_preview_never_follows_symlink(tmp_path: Path) -> None:
    original = tmp_path / "original"
    _frame(original)
    entry = tmp_path / "previews"
    entry.mkdir()
    try:
        (entry / ("a" * 64)).symlink_to(original / ("a" * 64), target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation requires Windows privilege")
    assert gallery._preview(entry, "a" * 64) is None


def test_command_creates_only_html_and_browser_is_optional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    saved = _saved()
    calls: list[str] = []
    monkeypatch.setattr(gallery, "show_pack_command", lambda selector: saved)
    monkeypatch.setattr(gallery.webbrowser, "open", lambda uri: calls.append(uri) or True)
    result = gallery.gallery_command("NewsEmoji", open_browser=False)
    path = Path(result.result["gallery_path"])
    assert path.parent == tmp_path and path.suffix == ".html"
    assert not calls and not result.result["browser_opened"]
    assert "Мультяшный взрыв" in path.read_text(encoding="utf-8")
    assert list(tmp_path.iterdir()) == [path]
    opened = gallery.gallery_command("NewsEmoji")
    assert calls == [Path(opened.result["gallery_path"]).as_uri()]
    assert opened.result["browser_opened"]


def test_browser_failure_retains_usable_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(gallery, "show_pack_command", lambda selector: _saved())

    def unavailable(uri: str) -> bool:
        raise OSError("no browser")

    monkeypatch.setattr(gallery.webbrowser, "open", unavailable)
    result = gallery.gallery_command("NewsEmoji")
    assert Path(result.result["gallery_path"]).is_file()
    assert not result.result["browser_opened"] and result.warnings


def test_previews_are_selected_by_exact_run_and_checkpoint_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MojiLexConfig(runs_dir=tmp_path / "runs", cache_dir=tmp_path / "cache")
    checkpoint = _save(
        config,
        elements={
            "123": ElementCheckpoint(stage="media_verified", source_descriptor_sha256="a" * 64)
        },
    )
    assert config.cache_dir is not None
    root = (
        config.cache_dir / "resume-media" / hashlib.sha256(checkpoint.run_id.encode()).hexdigest()
    )
    data = _frame(root)
    monkeypatch.setattr(gallery, "load_config", lambda: config)
    selected: list[tuple[str, str]] = []

    def resolve(selector: str, *, purpose: str):  # type: ignore[no-untyped-def]
        selected.append((selector, purpose))
        return checkpoint

    monkeypatch.setattr(gallery, "resolve_pack_run", resolve)
    before = _state(tmp_path)
    assert gallery._previews(checkpoint.run_id, {"123", "999"}) == {"123": data}
    assert selected == [(checkpoint.run_id, "view")]
    assert _state(tmp_path) == before
    monkeypatch.setattr(gallery, "_MAX_GALLERY_PREVIEW_BYTES", 1)
    assert gallery._previews(checkpoint.run_id, {"123"}) == {}


def test_saved_pack_gallery_preserves_staging_and_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MojiLexConfig(runs_dir=tmp_path / "runs", cache_dir=tmp_path / "cache")
    monkeypatch.setattr(packs, "load_config", lambda: config)
    monkeypatch.setattr(gallery, "load_config", lambda: config)
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(output))
    staging = tmp_path / "staging"
    snapshot = write_fixture(staging)
    checkpoint = _save(
        config,
        names=("SuspiciousCats",),
        extra={"staging_repository": str(staging)},
        elements={NATIVE_EMOJI_ID: ElementCheckpoint(stage="ai_facets_ready")},
    )
    before = _state(tmp_path)
    result = gallery.gallery_command("SuspiciousCats", open_browser=False)
    assert result.run_id == checkpoint.run_id
    assert result.result["counts"]["ready"] == 1
    path = Path(result.result["gallery_path"])
    page = path.read_text(encoding="utf-8")
    assert next(iter(snapshot.emojis.values())).descriptions["ru"].text in page
    after = _state(tmp_path)
    del after[str(path.relative_to(tmp_path))]
    assert before == after
