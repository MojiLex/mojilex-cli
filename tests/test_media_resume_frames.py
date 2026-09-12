from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from mojilex_cli.media.models import MediaLimitError, MediaLimits, MediaMetadata, ProcessedMedia
from mojilex_cli.media.resume import RetainedMediaStore
from mojilex_cli.media.temporary import TemporaryMediaRun

KEY = hashlib.sha256(b"safe source descriptor").hexdigest()


def _media(tmp_path: Path, *, dark: bool = True) -> ProcessedMedia:
    light = tmp_path / "generated-light.png"
    Image.new("RGB", (256, 256), "white").save(light)
    dark_paths: tuple[Path, ...] = ()
    if dark:
        black = tmp_path / "generated-dark.png"
        Image.new("RGB", (256, 256), "black").save(black)
        dark_paths = (black,)
    return ProcessedMedia(
        metadata=MediaMetadata(
            kind="static",
            format="webp",
            mime_type="image/webp",
            sha256="a" * 64,
            byte_size=100,
            width=512,
            height=512,
            animated=False,
        ),
        frame_paths=(light,),
        dark_frame_paths=dark_paths,
        rendered_frame_count=1,
        has_dark_render=dark,
    )


def _expected(media: ProcessedMedia) -> ProcessedMedia:
    return ProcessedMedia(
        metadata=media.metadata,
        analysis=media.analysis,
        frame_paths=(),
        rendered_frame_count=media.semantic_frame_count,
        has_dark_render=media.semantic_has_dark_render,
    )


def test_roundtrip_survives_removing_original_frames(tmp_path: Path) -> None:
    media = _media(tmp_path)
    root = tmp_path / "retained"
    store = RetainedMediaStore(root, 1_000_000)
    assert store.put(KEY, media)
    size = store.size_bytes
    assert size == sum(path.stat().st_size for path in (root / KEY).iterdir())
    expected_bytes = [path.read_bytes() for path in (*media.frame_paths, *media.dark_frame_paths)]
    for path in (*media.frame_paths, *media.dark_frame_paths):
        path.unlink()
    reopened = RetainedMediaStore(root, 1_000_000)
    assert reopened.size_bytes == size
    restored = reopened.get(KEY, _expected(media))
    assert restored is not None
    assert restored.metadata == media.metadata
    assert restored.semantic_frame_count == 1
    assert restored.semantic_has_dark_render
    assert [path.read_bytes() for path in (*restored.frame_paths, *restored.dark_frame_paths)] == (
        expected_bytes
    )
    assert all(
        path.suffix == ".png" for path in (*restored.frame_paths, *restored.dark_frame_paths)
    )


@pytest.mark.parametrize(
    "change", ["hash", "fingerprint", "traversal", "count", "bool_size", "larger_size"]
)
def test_invalid_manifest_is_a_miss_without_deletion(tmp_path: Path, change: str) -> None:
    media = _media(tmp_path)
    store = RetainedMediaStore(tmp_path / "retained", 1_000_000)
    assert store.put(KEY, media)
    path = store.root / KEY / "manifest.json"
    manifest = json.loads(path.read_bytes())
    if change == "hash":
        manifest["frames"][0]["sha256"] = "0" * 64
    elif change == "fingerprint":
        manifest["fingerprint"] = "0" * 64
    elif change == "traversal":
        manifest["frames"][0]["name"] = "../../outside.png"
    elif change == "count":
        manifest["frames"].pop()
    elif change == "bool_size":
        manifest["frames"][0]["bytes"] = True
    else:
        manifest["frames"][0]["bytes"] += 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    changed_bytes = path.read_bytes()
    assert store.get(KEY, _expected(media)) is None
    assert not store.put(KEY, media)
    assert path.read_bytes() == changed_bytes


def test_changed_metadata_count_dark_and_frame_bytes_are_misses(tmp_path: Path) -> None:
    media = _media(tmp_path)
    store = RetainedMediaStore(tmp_path / "retained", 1_000_000)
    assert store.put(KEY, media)
    expected = _expected(media)
    for changed in (
        expected.model_copy(
            update={"metadata": media.metadata.model_copy(update={"sha256": "b" * 64})}
        ),
        expected.model_copy(update={"rendered_frame_count": 2}),
        expected.model_copy(update={"has_dark_render": False}),
    ):
        assert store.get(KEY, changed) is None
    frame = store.root / KEY / "light-00.png"
    data = frame.read_bytes()
    frame.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    assert store.get(KEY, expected) is None


def test_store_limit_and_reservation_before_writing(tmp_path: Path) -> None:
    media = _media(tmp_path)
    store = RetainedMediaStore(tmp_path / "tiny", 1)
    assert not store.put(KEY, media)
    assert store.size_bytes == 0
    assert list(store.root.iterdir()) == []
    charged = []

    def reserve(size: int) -> None:
        assert list((tmp_path / "normal").iterdir()) == []
        charged.append(size)

    normal = RetainedMediaStore(tmp_path / "normal", 1_000_000, reserve=reserve)
    assert normal.put(KEY, media)
    assert charged == [normal.size_bytes]
    assert normal.put(KEY, media)
    assert len(charged) == 1
    exact = RetainedMediaStore(normal.root, normal.size_bytes)
    assert not exact.put("b" * 64, media)


def test_failed_write_releases_reservation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media = _media(tmp_path)
    charged: list[int] = []
    released: list[int] = []
    store = RetainedMediaStore(
        tmp_path / "retained",
        1_000_000,
        reserve=charged.append,
        release=released.append,
    )

    def fail(path: Path, data: bytes) -> None:
        path.write_bytes(data[:10])
        raise OSError("simulated full disk")

    monkeypatch.setattr(store, "_write", fail)
    assert not store.put(KEY, media)
    assert charged and charged == released
    assert store.size_bytes == 0
    assert list(store.root.iterdir()) == []


def test_reservation_limit_error_propagates_without_writing(tmp_path: Path) -> None:
    media = _media(tmp_path)

    def reserve(_: int) -> None:
        raise MediaLimitError("combined disk limit exceeded")

    store = RetainedMediaStore(tmp_path / "retained", 1_000_000, reserve=reserve)
    with pytest.raises(MediaLimitError):
        store.put(KEY, media)
    assert list(store.root.iterdir()) == []


def test_invalid_keys_unknown_entries_and_non_png_are_preserved(tmp_path: Path) -> None:
    media = _media(tmp_path, dark=False)
    store = RetainedMediaStore(tmp_path / "retained", 1_000_000)
    assert not store.put("../outside", media)
    assert store.get("../outside", _expected(media)) is None
    unknown = store.root / KEY
    unknown.mkdir()
    (unknown / "personal.txt").write_text("keep")
    assert not store.put(KEY, media)
    assert (unknown / "personal.txt").read_text() == "keep"
    media.frame_paths[0].write_bytes(b"raw source")
    assert not store.put("b" * 64, media)


def test_frame_and_root_symlinks_are_rejected(tmp_path: Path) -> None:
    media = _media(tmp_path)
    store = RetainedMediaStore(tmp_path / "retained", 1_000_000)
    assert store.put(KEY, media)
    frame = store.root / KEY / "light-00.png"
    outside = tmp_path / "outside.png"
    outside.write_bytes(frame.read_bytes())
    frame.unlink()
    try:
        frame.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation requires Windows developer mode or privileges")
    assert store.get(KEY, _expected(media)) is None
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(store.root, target_is_directory=True)
    linked_store = RetainedMediaStore(linked_root, 1_000_000)
    assert not linked_store.put("b" * 64, media)
    assert linked_store.get(KEY, _expected(media)) is None
    assert outside.read_bytes() == media.frame_paths[0].read_bytes()


def test_concurrent_same_entry_only_charges_once(tmp_path: Path) -> None:
    media = _media(tmp_path)
    charged: list[int] = []
    store = RetainedMediaStore(tmp_path / "retained", 1_000_000, reserve=charged.append)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: store.put(KEY, media), range(16)))
    assert all(results)
    assert charged == [store.size_bytes]


@pytest.mark.parametrize(
    "payload", [b" " * (32 * 1024 + 1), b"[" * 2000 + b"]" * 2000], ids=["oversize", "deep"]
)
def test_bounded_or_deeply_nested_corrupt_manifest_is_miss(
    tmp_path: Path,
    payload: bytes,
) -> None:
    media = _media(tmp_path)
    store = RetainedMediaStore(tmp_path / "retained", 1_000_000)
    assert store.put(KEY, media)
    (store.root / KEY / "manifest.json").write_bytes(payload)
    assert store.get(KEY, _expected(media)) is None


def test_oversized_frame_and_wrong_canvas_are_not_retained(tmp_path: Path) -> None:
    media = _media(tmp_path, dark=False)
    store = RetainedMediaStore(tmp_path / "retained", 30_000_000)
    with media.frame_paths[0].open("wb") as stream:
        stream.truncate(20 * 1024 * 1024 + 1)
    assert not store.put(KEY, media)
    Image.new("RGB", (512, 512), "white").save(media.frame_paths[0])
    assert not store.put(KEY, media)
    assert store.size_bytes == 0


def test_abandoned_partial_bytes_are_counted_and_preserved(tmp_path: Path) -> None:
    root = tmp_path / "retained"
    abandoned = root / ".pending-abandoned"
    abandoned.mkdir(parents=True)
    partial = abandoned / "light-00.png"
    partial.write_bytes(b"interrupted PNG write")
    store = RetainedMediaStore(root, partial.stat().st_size)
    assert store.size_bytes == partial.stat().st_size
    assert not store.put(KEY, _media(tmp_path))
    assert partial.read_bytes() == b"interrupted PNG write"


async def _chunks(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_reopened_retention_and_download_share_real_temporary_budget(tmp_path: Path) -> None:
    root = tmp_path / "retained"
    with TemporaryMediaRun(root=tmp_path) as original:
        assert original.path is not None
        media = _media(original.path)
        original.account_outputs((*media.frame_paths, *media.dark_frame_paths))
        store = RetainedMediaStore(
            root,
            100_000,
            reserve=original.reserve_retained_bytes,
            release=original.release_retained_bytes,
        )
        assert store.put(KEY, media)
        retained_size = store.size_bytes
    assert not media.frame_paths[0].exists()
    with TemporaryMediaRun(
        root=tmp_path,
        limits=MediaLimits(max_run_temp_bytes=retained_size + 5),
    ) as resumed:
        reopened = RetainedMediaStore(
            root,
            resumed.limits.max_run_temp_bytes,
            reserve=resumed.reserve_retained_bytes,
            release=resumed.release_retained_bytes,
        )
        resumed.reserve_retained_bytes(reopened.size_bytes)
        assert reopened.get(KEY, _expected(media)) is not None
        assert resumed.path is not None
        with pytest.raises(MediaLimitError, match="run temporary disk limit"):
            await resumed.write_stream(_chunks(b"123", b"456"))
        assert list(resumed.path.iterdir()) == []
        assert resumed.bytes_written == 0
        source, _, size = await resumed.write_stream(_chunks(b"12345"))
        assert source.read_bytes() == b"12345"
        assert size == 5
        with pytest.raises(MediaLimitError):
            resumed.reserve_retained_bytes(1)


@pytest.mark.asyncio
async def test_failed_retention_releases_real_temporary_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TemporaryMediaRun(
        root=tmp_path,
        limits=MediaLimits(max_run_temp_bytes=5000),
    ) as temporary:
        assert temporary.path is not None
        media = _media(temporary.path)
        temporary.account_outputs((*media.frame_paths, *media.dark_frame_paths))
        store = RetainedMediaStore(
            tmp_path / "retained",
            temporary.limits.max_run_temp_bytes,
            reserve=temporary.reserve_retained_bytes,
            release=temporary.release_retained_bytes,
        )

        def fail(path: Path, data: bytes) -> None:
            path.write_bytes(data[:10])
            raise OSError("simulated full disk")

        monkeypatch.setattr(store, "_write", fail)
        assert not store.put(KEY, media)
        assert store.size_bytes == 0
        assert list(store.root.iterdir()) == []
        remaining = temporary.limits.max_run_temp_bytes - temporary.bytes_written
        source, _, size = await temporary.write_stream(_chunks(b"x" * remaining))
        assert source.stat().st_size == size == remaining
        assert temporary.bytes_written == temporary.limits.max_run_temp_bytes
