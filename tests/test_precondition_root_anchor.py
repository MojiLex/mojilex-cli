from collections import Counter
from pathlib import Path, PurePosixPath

import pytest

from mojilex_cli.dataset import layout, transaction


def _baseline(root: Path) -> dict[PurePosixPath, bytes]:
    expected = {PurePosixPath("dataset.json"): b"{}\n"}
    expected.update(
        (PurePosixPath(f"data/telegram/emojis/{index:06x}.jsonl"), b"old\n") for index in range(32)
    )
    for relative, content in expected.items():
        target = root.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return expected


def test_precondition_resolves_root_once_and_each_target_fresh(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    expected = _baseline(root)
    resolutions = Counter()
    checks = Counter()
    hashes = Counter()
    real_resolve = Path.resolve
    real_check = layout.is_link_or_reparse_point
    real_hash = transaction._file_sha256

    def resolve(path, *args, **kwargs):
        resolutions[path] += 1
        return real_resolve(path, *args, **kwargs)

    def check(path):
        checks[path] += 1
        return real_check(path)

    def file_hash(path):
        hashes[path] += 1
        return real_hash(path)

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(layout, "is_link_or_reparse_point", check)
    monkeypatch.setattr(transaction, "_file_sha256", file_hash)
    transaction._verify_dataset_precondition(root, expected)

    assert resolutions[root] == 1
    for relative in expected:
        target = root.joinpath(*relative.parts)
        assert resolutions[target] == 1
        assert checks[target] >= 1
        assert hashes[target] == 1
    assert checks[root] >= len(expected)
    assert checks[root / "data" / "telegram" / "emojis"] >= len(expected) - 1


@pytest.mark.parametrize("component", ["root", "ancestor", "target"])
def test_precondition_anchor_rejects_reparse_introduced_between_reads(
    tmp_path, monkeypatch, component
):
    root = tmp_path.resolve()
    expected = _baseline(root)
    last_target = root.joinpath(*next(reversed(expected)).parts)
    changed = {
        "root": root,
        "ancestor": last_target.parent,
        "target": last_target,
    }[component]
    real_check = layout.is_link_or_reparse_point
    real_hash = transaction._file_sha256
    introduced = False
    hashed = []

    def check(path):
        return (introduced and path == changed) or real_check(path)

    def file_hash(path):
        nonlocal introduced
        hashed.append(path)
        result = real_hash(path)
        introduced = True
        return result

    monkeypatch.setattr(layout, "is_link_or_reparse_point", check)
    monkeypatch.setattr(transaction, "_file_sha256", file_hash)
    with pytest.raises(transaction.AtomicWriteError, match="unsafe path"):
        transaction._verify_dataset_precondition(root, expected)
    assert last_target not in hashed


def test_resolved_root_anchor_cannot_redirect_to_another_tree(tmp_path):
    root = tmp_path.resolve()
    other = root / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="absolute lexical path"):
        layout.safe_destination(root, "dataset.json", _resolved_root=other)
    with pytest.raises(ValueError, match="absolute lexical path"):
        layout.safe_destination(root, "dataset.json", _resolved_root=Path("other"))


def test_resolved_root_anchor_still_resolves_target_containment(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    target = root / "dataset.json"
    target.write_bytes(b"{}\n")
    real_resolve = Path.resolve

    def redirected(path, *args, **kwargs):
        if path == target:
            return root.parent / "outside.json"
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", redirected)
    with pytest.raises(ValueError, match="escapes dataset root"):
        layout.safe_destination(root, "dataset.json", canonical=True, _resolved_root=root)
