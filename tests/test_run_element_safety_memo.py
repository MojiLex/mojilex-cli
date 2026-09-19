import pytest

from mojilex_cli.runs import RunStore, RunStoreError, new_checkpoint
from mojilex_cli.runs import store as store_module


def checkpoint_with_element():
    checkpoint = new_checkpoint(
        command="import",
        safe_parameters={},
        cli_version="0.2.0",
        schema_version="1.0.0",
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    )
    checkpoint.elements["item"] = store_module.ElementCheckpoint(stage="discovered")
    return checkpoint


def test_element_memo_reuses_exact_content_and_rechecks_changed_copy(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "runs")
    checkpoint = checkpoint_with_element()
    path = store.save(checkpoint)
    original = path.read_bytes()
    calls = []
    previous = store_module._assert_safe

    def tracked(value, *, path="checkpoint", memo=None):
        if path == "checkpoint.elements.item" and isinstance(value, dict):
            calls.append(value)
        return previous(value, path=path, memo=memo)

    monkeypatch.setattr(store_module, "_assert_safe", tracked)
    checkpoint.elements["item"] = checkpoint.elements["item"].model_copy(deep=True)
    store.save(checkpoint)
    assert not calls
    checkpoint.elements["item"] = checkpoint.elements["item"].model_copy(
        update={"error_code": "https://user:password@example.test/"}
    )
    with pytest.raises(RunStoreError, match="credential"):
        store.save(checkpoint)
    assert len(calls) == 1
    assert path.read_bytes() == original


def test_safe_element_does_not_authorize_unsafe_element_key(tmp_path):
    store = RunStore(tmp_path / "runs")
    checkpoint = checkpoint_with_element()
    store.save(checkpoint)
    checkpoint.elements["download_url"] = checkpoint.elements.pop("item")
    with pytest.raises(RunStoreError, match="unsafe field"):
        store.save(checkpoint)


def test_element_memo_invalidates_after_detector_change(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "runs")
    checkpoint = checkpoint_with_element()
    store.save(checkpoint)
    assert store._safe_payloads.entries
    previous = store_module.contains_secret_text
    monkeypatch.setattr(
        store_module, "contains_secret_text", lambda text: text == "discovered" or previous(text)
    )
    with pytest.raises(RunStoreError, match="credential"):
        store.save(checkpoint)
    assert not store._safe_payloads.entries


def test_element_memo_eviction_and_oversized_entries(monkeypatch):
    monkeypatch.setattr(store_module._SafePayloadMemo, "MAX_BYTES", 12)
    monkeypatch.setattr(store_module._SafePayloadMemo, "MAX_ENTRIES", 2)
    memo = store_module._SafePayloadMemo()
    memo.remember(b"aaaa")
    memo.remember(b"bbbb")
    assert memo.contains(b"aaaa")
    memo.remember(b"cccc")
    assert not memo.contains(b"bbbb")
    memo.remember(b"d" * 10)
    assert memo.size_bytes == 10
    assert len(memo.entries) == 1
    memo.remember(b"e" * 13)
    assert memo.size_bytes == 10
    memo.clear()
    assert not memo.entries and memo.size_bytes == 0


def test_changed_nested_dict_in_unvalidated_copy_is_rechecked(tmp_path):
    store = RunStore(tmp_path / "runs")
    checkpoint = checkpoint_with_element()
    # model_copy deliberately bypasses Pydantic validation; safety must still
    # inspect nested mutable content rather than trust the frozen outer model.
    nested = {"note": "ordinary"}
    checkpoint.elements["item"] = checkpoint.elements["item"].model_copy(
        update={"error_code": nested}
    )
    with pytest.warns(UserWarning, match="Pydantic"):
        path = store.save(checkpoint)
    original = path.read_bytes()
    nested["api_key"] = "placeholder"
    with pytest.warns(UserWarning, match="Pydantic"):
        with pytest.raises(RunStoreError, match="unsafe field"):
            store.save(checkpoint)
    assert path.read_bytes() == original
