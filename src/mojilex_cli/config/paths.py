"""Persistent per-user application storage, independent of the launch directory."""

from pathlib import Path

from platformdirs import user_state_path


def default_workspace_dir() -> Path:
    """Return application data without creating directories."""

    return user_state_path("mojilex", "MojiLex").expanduser().absolute()


def default_repository_path() -> Path:
    """Return the managed local checkout of the public dataset."""

    return default_workspace_dir() / "repository"
