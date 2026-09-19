from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from mojilex_cli.media import resume
from mojilex_cli.media.models import ProcessedMedia
from mojilex_cli.media.resume import RetainedMediaStore
from test_media_resume_frames import KEY, _expected, _media


def _saved(tmp_path: Path) -> tuple[RetainedMediaStore, ProcessedMedia]:
    media = _media(tmp_path)
    tile = tmp_path / "tile.png"
    Image.new("RGBA", (100, 100), (10, 20, 30, 200)).save(tile)
    media = media.model_copy(
        update={
            "metadata": media.metadata.model_copy(update={"width": 100, "height": 100}),
            "composition_tile_path": tile,
            "composition_tile_sha256": hashlib.sha256(tile.read_bytes()).hexdigest(),
        }
    )
    store = RetainedMediaStore(tmp_path / "retained", 1_000_000)
    assert store.put(KEY, media)
    return store, _expected(media)


def test_tile_only_restoration_never_reads_semantic_frames(tmp_path, monkeypatch) -> None:
    store, expected = _saved(tmp_path)
    reads = []
    original = resume._read

    def read(path, limit):
        reads.append(path.name)
        assert path.name in {"manifest.json", "composition-tile.png"}
        return original(path, limit)

    monkeypatch.setattr(resume, "_read", read)
    restored = store.get_composition_tile(KEY, expected)
    assert restored is not None
    assert reads == ["manifest.json", "composition-tile.png"]
    assert restored.frame_paths == restored.dark_frame_paths == ()
    assert restored.semantic_frame_count == expected.semantic_frame_count
    assert restored.semantic_has_dark_render == expected.semantic_has_dark_render
    assert restored.metadata == expected.metadata
    assert restored.analysis == expected.analysis
    assert restored.composition_tile_path == store.root / KEY / "composition-tile.png"


def test_valid_tile_is_independent_of_missing_or_corrupt_semantic_frames(tmp_path) -> None:
    store, expected = _saved(tmp_path)
    (store.root / KEY / "light-00.png").unlink()
    (store.root / KEY / "dark-00.png").write_bytes(b"broken")
    assert store.get(KEY, expected) is None
    assert store.get_composition_tile(KEY, expected) is not None


@pytest.mark.parametrize(
    "change",
    ["missing", "hash", "size", "bool_size", "name", "fingerprint", "frame_count", "version"],
)
def test_tile_manifest_mismatch_is_a_nonmutating_cache_miss(tmp_path, change) -> None:
    store, expected = _saved(tmp_path)
    path = store.root / KEY / "manifest.json"
    manifest = json.loads(path.read_bytes())
    if change == "missing":
        del manifest["composition_tile"]
    elif change == "fingerprint":
        manifest["fingerprint"] = "0" * 64
    elif change == "frame_count":
        manifest["frames"].pop()
    elif change == "version":
        manifest["version"] = True
    else:
        tile = manifest["composition_tile"]
        if change == "hash":
            tile["sha256"] = "0" * 64
        elif change == "size":
            tile["bytes"] += 1
        elif change == "bool_size":
            tile["bytes"] = True
        else:
            tile["name"] = "../tile.png"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    before = path.read_bytes()
    assert store.get_composition_tile(KEY, expected) is None
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["hash", "bad_png", "wrong_size", "rgb", "missing"])
def test_invalid_tile_bytes_are_rejected_even_with_rewritten_hash(tmp_path, change) -> None:
    store, expected = _saved(tmp_path)
    path = store.root / KEY / "composition-tile.png"
    manifest_path = store.root / KEY / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if change == "missing":
        path.unlink()
    else:
        if change in {"hash", "bad_png"}:
            path.write_bytes(b"not a PNG")
        else:
            Image.new(
                "RGB" if change == "rgb" else "RGBA",
                (90, 100) if change == "wrong_size" else (100, 100),
            ).save(path)
        if change != "hash":
            data = path.read_bytes()
            manifest["composition_tile"].update(
                bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert store.get_composition_tile(KEY, expected) is None


@pytest.mark.parametrize("field,value", [("sha256", "b" * 64), ("width", 99), ("byte_size", 101)])
def test_tile_requires_exact_expected_metadata(tmp_path, field, value) -> None:
    store, expected = _saved(tmp_path)
    changed = expected.model_copy(
        update={"metadata": expected.metadata.model_copy(update={field: value})}
    )
    assert store.get_composition_tile(KEY, changed) is None
    assert store.get_composition_tile("../invalid-key", expected) is None
