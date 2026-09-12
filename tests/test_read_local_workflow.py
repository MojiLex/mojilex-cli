from __future__ import annotations

import copy
import json

import pytest
from typer.testing import CliRunner

from mojilex_cli.cli import app
from mojilex_cli.commands import read
from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.config import MojiLexConfig
from mojilex_cli.i18n import use_ui_language
from mojilex_cli.read import local
from mojilex_cli.read.service import SnapshotReader
from mojilex_cli.read.snapshot import LoadedSnapshot, validate_embedded_schema_instance
from test_read_access import FakeSnapshot


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MOJILEX_SNAPSHOT", raising=False)
    monkeypatch.setattr(local, "load_config", lambda: MojiLexConfig())
    return tmp_path


def _manifest(path, suffix="1"):
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps({"dataset": "mojilex", "snapshot_id": f"data-2026.09.13.{suffix}"}),
        encoding="utf-8",
    )
    return path


def test_discovery_unique_ambiguous_and_explicit_override(isolated, monkeypatch):
    first = _manifest(isolated / "dist")
    assert local.select_local_snapshot(None) == first
    second = _manifest(isolated / "snapshots" / "other", "2")
    with pytest.raises(CommandError) as error:
        local.select_local_snapshot(None)
    assert error.value.error.code == "OPTION_CONFLICT"
    assert local.select_local_snapshot(first) == first
    monkeypatch.setenv("MOJILEX_SNAPSHOT", str(second))
    assert local.select_local_snapshot(None) == second


def test_discovery_uses_configured_repository_but_not_unrelated_subdirectories(
    isolated, monkeypatch
):
    repo = isolated / "configured"
    expected = _manifest(repo / "dist")
    _manifest(isolated / "unrelated" / "private")
    monkeypatch.setattr(
        local, "load_config", lambda: MojiLexConfig(repository={"target": str(repo)})
    )
    assert [item.path for item in local.discover_local_snapshots()] == [expected]


def test_snapshots_empty_is_actionable_success_and_get_no_longer_requires_option(isolated):
    result = CliRunner().invoke(app, ["snapshots", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["result"]["items"] == []
    assert any(item["code"] == "NO_LOCAL_SNAPSHOTS" for item in payload["warnings"])
    missing = CliRunner().invoke(app, ["get", "5210956306952758910", "--json"])
    assert json.loads(missing.stdout)["errors"][0]["code"] == "SNAPSHOT_NOT_FOUND"


def test_snapshots_lists_paths_with_schema_valid_json_and_pagination(isolated):
    _manifest(isolated / "snapshots" / "first")
    _manifest(isolated / "snapshots" / "second", "2")
    result = CliRunner().invoke(app, ["snapshots", "--json", "--limit", "1"])
    payload = json.loads(result.stdout)
    validate_embedded_schema_instance(
        payload,
        "mlx://schemas/distribution/v1/cli-read-envelope.schema.json",
        location="test",
        invalid_code="INTERNAL_ERROR",
    )
    assert payload["result"]["items"][0]["catalog_status"] == "unknown"
    assert any("--snapshot" in item["message"] for item in payload["warnings"])
    cursor = payload["result"]["next_cursor"]["value"]
    second = CliRunner().invoke(app, ["snapshots", "--json", "--limit", "1", "--cursor", cursor])
    assert json.loads(second.stdout)["result"]["items"][0]["snapshot_id"].endswith(".2")
    human = CliRunner().invoke(app, ["snapshots"])
    assert str(isolated / "snapshots" / "first") in human.stdout


def test_auto_discovery_keeps_unsigned_snapshot_opt_in_gate(isolated, monkeypatch):
    path = _manifest(isolated / "dist")
    gate = LoadedSnapshot(
        root=path,
        manifest_path=path / "manifest.json",
        manifest={"snapshot_id": "data-2026.09.13.1", "trust_stage": "pre-enforcement"},
        manifest_sha256="a" * 64,
        descriptors={},
        verified_paths={},
    )
    monkeypatch.setattr(read, "load_snapshot", lambda *args, **kwargs: gate)
    with pytest.raises(CommandError) as error:
        read._open({}, None, None, False, True)
    assert error.value.error.code == "SIGNATURE_MISSING"


def test_get_accepts_exact_native_telegram_id_without_changing_canonical_output():
    snapshot = FakeSnapshot()
    reader = SnapshotReader(snapshot)
    emoji = snapshot._rows["emojis"][0]
    assert reader.get(
        emoji["native_id"], view="canonical", language=None, include_sensitive=False
    ) == reader.get(emoji["id"], view="canonical", language=None, include_sensitive=False)


def test_get_native_id_does_not_choose_an_ambiguous_epoch():
    snapshot = FakeSnapshot()
    original = snapshot._rows["emojis"][0]
    other = copy.deepcopy(original)
    other.update(id="mxe_44444444-4444-5444-8444-444444444444", identity_epoch=1)
    snapshot._rows["emojis"] += (other,)
    with pytest.raises(CommandError) as error:
        SnapshotReader(snapshot).get(
            original["native_id"], view="canonical", language=None, include_sensitive=False
        )
    assert error.value.error.code == "AMBIGUOUS_REFERENCE"


def test_search_native_id_preserves_safe_agent_visibility():
    snapshot = FakeSnapshot()
    reader = SnapshotReader(snapshot)
    kwargs = dict(
        query="9007199254740993",
        language="en",
        filters=read._validate_filters({}),
        sort="relevance",
        limit=20,
        cursor=None,
    )
    assert len(reader.search(view="search", **kwargs)["items"]) == 1
    assert reader.search(view="agent", **kwargs)["items"] == []


def test_explicit_snapshot_discovery_does_not_require_authoring_config(isolated, monkeypatch):
    selected = _manifest(isolated / "pinned")
    monkeypatch.setenv("MOJILEX_SNAPSHOT", str(selected))

    def unavailable_config():
        raise PermissionError("unrelated authoring config is unavailable")

    monkeypatch.setattr(local, "load_config", unavailable_config)
    assert local.discover_local_snapshots()[0].path == selected


def test_human_russian_get_is_readable_but_json_payload_stays_exact(capsys):
    result = read.ReadCommandResult(
        result={
            "item": {
                "record": {
                    "id": "mxe_example",
                    "native_id": "123456789",
                    "descriptions": {"ru": {"text": "Радостный кот"}},
                }
            },
            "view": "canonical",
        },
        dataset=None,
        arguments={},
        view="canonical",
        warnings=[{"code": "TEST_WARNING", "message": "Diagnostic warning"}],
    )
    with use_ui_language("ru"):
        read.execute_read("get", lambda: result, json_output=False)
        human = capsys.readouterr().out
        read.execute_read("get", lambda: result, json_output=True)
        machine = json.loads(capsys.readouterr().out)
    assert "Радостный кот" in human and "123456789" in human
    assert "Diagnostic warning" in human and "\\u0420" not in human
    assert machine["result"] == result.result


def test_machine_discovery_messages_remain_english_under_russian_ui(isolated, capsys):
    with use_ui_language("ru"):
        read.snapshots_cli(json_output=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["warnings"][0]["message"].startswith("Local files only")


def test_empty_unsigned_agent_view_explains_diagnostic_option(capsys):
    result = read.ReadCommandResult(
        result={"items": []},
        dataset={"release_verification_status": "integrity-only-unsigned"},
        arguments={},
        view="agent",
    )
    with use_ui_language("ru"):
        read.execute_read("search", lambda: result, json_output=False)
    text = capsys.readouterr().out
    assert "--view canonical --allow-unverified" in text
    assert "не подтверждает доверие" in text
