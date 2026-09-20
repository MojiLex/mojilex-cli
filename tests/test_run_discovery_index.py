import json
import os

import pytest

from mojilex_cli.commands.import_reuse import reusable_imports
from mojilex_cli.config import MojiLexConfig
from mojilex_cli.runs import RunStore, new_checkpoint
from mojilex_cli.runs.index import read_index


def saved(tmp_path, name="Alpha"):
    staging = tmp_path / "staging"
    staging.mkdir(exist_ok=True)
    checkpoint = new_checkpoint(
        command="import",
        cli_version="0.1.0",
        schema_version="1.0",
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
        safe_parameters={
            "sources": [f"https://t.me/addemoji/{name}"],
            "staging_repository": str(staging),
            "max_items": None,
        },
    )
    store = RunStore(tmp_path / "runs")
    store.save(checkpoint)
    return store, checkpoint


def test_warm_discovery_deserializes_only_requested_winners(tmp_path, monkeypatch):
    store, alpha = saved(tmp_path)
    _, beta = saved(tmp_path, "Beta")
    loads = []
    original = RunStore.load

    def tracked(self, run_id):
        loads.append(run_id)
        return original(self, run_id)

    monkeypatch.setattr(RunStore, "load", tracked)
    config = MojiLexConfig(repository={"target": "MojiLex/mojilex"}, runs_dir=store.root)
    result = reusable_imports(["Alpha"], config, max_items=None)
    assert result["Alpha"][0] == alpha
    assert loads == [alpha.run_id]
    assert beta.run_id not in loads


@pytest.mark.parametrize("damage", ["missing", "invalid", "version"])
def test_missing_or_corrupt_index_rebuilds_from_checkpoint(tmp_path, damage):
    store, checkpoint = saved(tmp_path)
    path = store.root / "indexes" / f"{checkpoint.run_id}.json"
    if damage == "missing":
        path.unlink()
    elif damage == "invalid":
        path.write_text("{")
    else:
        raw = json.loads(path.read_bytes())
        raw["version"] = 99
        path.write_text(json.dumps(raw))
    index = read_index(store, checkpoint.run_id)
    assert index.sources[0].name == "alpha"
    assert json.loads(path.read_bytes())["version"] == 1


def test_same_size_same_timestamp_checkpoint_edit_invalidates_index(tmp_path):
    store, checkpoint = saved(tmp_path)
    path = store.root / f"{checkpoint.run_id}.json"
    before = path.stat()
    payload = path.read_bytes().replace(b"/Alpha", b"/Bravo")
    path.write_bytes(payload)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert read_index(store, checkpoint.run_id).sources[0].name == "bravo"


def test_index_write_failure_preserves_checkpoint(tmp_path, monkeypatch):
    from mojilex_cli.runs import index

    store, checkpoint = saved(tmp_path)

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(index, "_atomic_write", fail)
    store.save(checkpoint)
    assert store.load(checkpoint.run_id) == checkpoint


def test_parsed_cache_isolated_from_mutated_checkpoint(tmp_path):
    store, checkpoint = saved(tmp_path)
    first = store.load(checkpoint.run_id)
    first.safe_parameters["sources"].append("https://t.me/addemoji/Changed")
    second = RunStore(store.root).load(checkpoint.run_id)
    assert second.safe_parameters["sources"] == ["https://t.me/addemoji/Alpha"]


def test_parsed_cache_invalidates_when_safety_rules_change(tmp_path, monkeypatch):
    from mojilex_cli.runs import store as module

    store, checkpoint = saved(tmp_path)
    store.load(checkpoint.run_id)
    monkeypatch.setattr(module, "contains_secret_text", lambda value: value == "Alpha")
    # Existing source text does not exactly match; the replacement still must run.
    seen = []
    monkeypatch.setattr(module, "contains_secret_text", lambda value: seen.append(value) or False)
    store.load(checkpoint.run_id)
    assert seen


def test_valid_json_summary_damage_cannot_hide_saved_pack(tmp_path):
    store, checkpoint = saved(tmp_path)
    path = store.root / "indexes" / f"{checkpoint.run_id}.json"
    data = json.loads(path.read_bytes())
    data["sources"][0]["name"] = "different"
    path.write_text(json.dumps(data))
    assert read_index(store, checkpoint.run_id).sources[0].name == "alpha"


def test_index_rebuild_parses_exact_bytes_without_second_checkpoint_read(tmp_path, monkeypatch):
    store, checkpoint = saved(tmp_path)
    (store.root / "indexes" / f"{checkpoint.run_id}.json").unlink()

    def unexpected(*args):
        raise AssertionError("checkpoint was reopened")

    monkeypatch.setattr(store, "load", unexpected)
    assert read_index(store, checkpoint.run_id).sources[0].name == "alpha"
