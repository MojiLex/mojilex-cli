import copy
import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from mojilex_cli.commands import gallery, packs
from mojilex_cli.config import MojiLexConfig
from mojilex_cli.runs import ElementCheckpoint
from test_gallery_command import _saved
from test_pack_commands import _save


def _group():
    return {
        "detector": "composition-v1",
        "rows": 2,
        "columns": 2,
        "verified": True,
        "verifier_model": "test",
        "members": [
            {"native_id": str(i), "media_sha256": "b" * 64, "tile_sha256": "c" * 64}
            for i in range(1, 5)
        ],
    }


def _checkpoint(tmp_path):
    config = MojiLexConfig(runs_dir=tmp_path / "runs", cache_dir=tmp_path / "cache")
    checkpoint = _save(
        config,
        elements={
            str(i): ElementCheckpoint(
                stage="media_verified",
                media_sha256=("b" * 64,),
                source_descriptor_sha256=str(i) * 64,
            )
            for i in range(1, 5)
        },
    )
    return config, checkpoint.model_copy(
        update={
            "safe_parameters": {
                **checkpoint.safe_parameters,
                "source_memberships": {"NewsEmoji": [str(i) for i in range(1, 5)]},
                "composition_evidence": {"NewsEmoji": [_group()]},
            }
        }
    )


def test_show_keeps_only_current_verified_group(tmp_path: Path) -> None:
    _, checkpoint = _checkpoint(tmp_path)
    assert packs._saved_compositions(checkpoint, ("NewsEmoji",)) == [
        {**_group(), "verification_passes": 0}
    ]
    assert packs._saved_compositions(checkpoint, ("OtherEmoji",)) == []


@pytest.mark.parametrize(
    "change",
    [
        "unverified",
        "stale",
        "membership",
        "duplicate",
        "overlap",
        "shape",
        "unknown",
        "coercion",
    ],
)
def test_show_rejects_unsafe_evidence(tmp_path: Path, change: str) -> None:
    _, checkpoint = _checkpoint(tmp_path)
    parameters = copy.deepcopy(checkpoint.safe_parameters)
    group = parameters["composition_evidence"]["NewsEmoji"][0]
    if change == "unverified":
        group["verified"] = False
    elif change == "stale":
        group["members"][0]["media_sha256"] = "f" * 64
    elif change == "membership":
        parameters["source_memberships"]["NewsEmoji"].pop()
    elif change == "duplicate":
        group["members"][0] = group["members"][1]
    elif change == "overlap":
        parameters["composition_evidence"]["NewsEmoji"].append(copy.deepcopy(group))
    elif change == "shape":
        group["columns"] = 3
    elif change == "unknown":
        group["injected"] = "x"
    else:
        group["verified"] = "true"
    checkpoint = checkpoint.model_copy(update={"safe_parameters": parameters})
    assert packs._saved_compositions(checkpoint, ("NewsEmoji",)) == []


@pytest.mark.parametrize("corrupt", [False, True])
def test_gallery_uses_only_complete_checksums_of_native_tiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt: bool,
) -> None:
    config, checkpoint = _checkpoint(tmp_path)
    assert config.cache_dir is not None
    root = (
        config.cache_dir / "resume-media" / hashlib.sha256(checkpoint.run_id.encode()).hexdigest()
    )
    group = _group()
    for member in group["members"]:
        entry = root / (member["native_id"] * 64)
        entry.mkdir(parents=True)
        stream = io.BytesIO()
        Image.new("RGBA", (100, 100), (10, 20, 30, 100)).save(stream, format="PNG")
        data = stream.getvalue()
        member["tile_sha256"] = hashlib.sha256(data).hexdigest()
        (entry / "composition-tile.png").write_bytes(data)
        (entry / "manifest.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "composition_tile": {
                        "name": "composition-tile.png",
                        "bytes": len(data),
                        "sha256": member["tile_sha256"],
                    },
                }
            )
        )
    if corrupt:
        (root / ("1" * 64) / "composition-tile.png").write_bytes(b"corrupt")
    monkeypatch.setattr(gallery, "resolve_pack_run", lambda *a, **kw: checkpoint)
    monkeypatch.setattr(gallery, "load_config", lambda: config)
    tiles = gallery._composition_tiles(checkpoint.run_id, [group])
    result = _saved()
    result.result["compositions"] = [group]
    before = copy.deepcopy(result.result["items"])
    document = gallery.render_gallery(result, {}, tiles)
    assert ("Связанные фрагменты изображения" in document) is not corrupt
    assert len(tiles) == (0 if corrupt else 4)
    assert result.result["items"] == before
    assert "Мультяшный взрыв" in document
    assert "Связанные фрагменты изображения" not in gallery.render_gallery(result, {"123": data})
