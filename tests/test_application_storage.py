from pathlib import Path

import pytest

from mojilex_cli.commands import system
from mojilex_cli.config import MojiLexConfig, load_config, loader, paths


def test_application_defaults_are_dynamic_and_do_not_create_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = tmp_path / "LocalAppData" / "MojiLex" / "mojilex"
    monkeypatch.setattr(paths, "user_state_path", lambda *_args: storage)
    monkeypatch.setattr(loader, "user_state_path", lambda *_args: storage)

    first = MojiLexConfig()
    assert first.repository.target == str(storage / "repository")
    assert loader.default_runs_dir() == storage / "runs"
    assert not storage.exists()

    another = tmp_path / "another-user" / "LocalAppData" / "MojiLex" / "mojilex"
    monkeypatch.setattr(paths, "user_state_path", lambda *_args: another)
    assert MojiLexConfig().repository.target == str(another / "repository")


def test_existing_history_keeps_its_default_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loader, "user_state_path", lambda *_args: tmp_path / "state")
    legacy = tmp_path / "state" / "runs"
    legacy.mkdir(parents=True)
    (legacy / "unrelated.json").write_text("{}", encoding="utf-8")
    assert loader.default_runs_dir() == legacy
    (legacy / ("mlxrun_" + "a" * 32 + ".json")).write_text("{}", encoding="utf-8")
    assert loader.default_runs_dir() == legacy


def test_configured_repository_and_storage_override_application_defaults(tmp_path: Path) -> None:
    user = tmp_path / "user.toml"
    user.write_text(
        'runs_dir = "chosen-runs"\n[repository]\ntarget = "Someone/dataset"\n',
        encoding="utf-8",
    )
    config = load_config(user_path=user, project_path=tmp_path / "absent", environment={})
    assert config.runs_dir == Path("chosen-runs")
    assert config.repository.target == "Someone/dataset"
    config = load_config(
        user_path=user,
        project_path=tmp_path / "absent",
        environment={"MOJILEX_RUNS_DIR": "env-runs", "MOJILEX_REPO": "Other/dataset"},
    )
    assert config.runs_dir == Path("env-runs")
    assert config.repository.target == "Other/dataset"


def test_unprovisioned_default_repository_has_official_github_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paths, "user_state_path", lambda *_args: tmp_path / "appdata")
    assert str(system._repository_ref(str(paths.default_repository_path()))) == "MojiLex/mojilex"
    assert system._repository_ref(str(tmp_path / "missing-custom-repository")) is None
    assert not paths.default_repository_path().exists()


def test_existing_default_directory_requires_real_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paths, "user_state_path", lambda *_args: tmp_path / "appdata")
    paths.default_repository_path().mkdir(parents=True)
    monkeypatch.setattr(system, "_run_git", lambda *_args, **_kwargs: None)
    assert system._repository_ref(str(paths.default_repository_path())) is None
