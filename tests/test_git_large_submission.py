import subprocess

import pytest

from mojilex_cli.git import GitError, GitRunner


def _repository(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    runner = GitRunner(tmp_path)
    runner.run("config", "user.name", "Test User")
    runner.run("config", "user.email", "test@example.invalid")
    return runner


def test_large_submission_stages_and_commits_exact_paths(tmp_path, monkeypatch):
    runner = _repository(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    paths = tuple(f"data/emoji-{index:05d}-{'a' * 40}.json" for index in range(7080))
    for path in paths:
        (tmp_path / path).write_text("{}\n", encoding="utf-8")
    (tmp_path / "unrelated.txt").write_text("keep outside the commit\n", encoding="utf-8")
    assert len(subprocess.list2cmdline(["git", "add", "--", *paths])) > 32767
    real_run = subprocess.run
    command_lengths = []

    def bounded_run(command, **kwargs):
        command_lengths.append(len(subprocess.list2cmdline(command)))
        assert command_lengths[-1] < 32767
        return real_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", bounded_run)
    assert runner.stage_paths(paths) == paths
    assert set(runner.staged_paths()) == set(paths)
    commit = runner.commit("data: large synthetic submission")
    assert set(runner.run("ls-tree", "-r", "--name-only", "-z", commit).stdout.split("\0")) == {
        *paths,
        "",
    }
    assert runner.status_paths() == ("unrelated.txt",)


def test_staging_preserves_literal_unicode_paths_and_deletions(tmp_path):
    runner = _repository(tmp_path)
    for name in ("emoji[1].json", "emoji1.json", "эмодзи 1.json", "delete.json"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    (tmp_path / "delete.json").write_text('{"old": true}\n', encoding="utf-8")
    runner.stage_paths(("delete.json",))
    runner.commit("data: initial")
    (tmp_path / "delete.json").unlink()
    targets = ("emoji[1].json", "эмодзи 1.json", "delete.json")
    runner.stage_paths(targets)
    assert set(runner.staged_paths()) == set(targets)
    runner.commit("data: exact paths")
    assert runner.status_paths() == ("emoji1.json",)


@pytest.mark.parametrize("path", ["../outside.json", "data/bad\0path.json", "data/bad\npath"])
def test_stdin_paths_keep_validation(tmp_path, path):
    runner = _repository(tmp_path)
    with pytest.raises(GitError):
        runner.stage_paths((path,))
    assert runner.staged_paths() == ()


@pytest.mark.parametrize("winerror, message", [(206, "length limit"), (2, "not found")])
def test_windows_launch_errors_are_distinguished(tmp_path, monkeypatch, winerror, message):
    runner = _repository(tmp_path)

    def fail(*args, **kwargs):
        error = FileNotFoundError("launch failed")
        error.winerror = winerror
        raise error

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(GitError, match=message):
        runner.run("status")
