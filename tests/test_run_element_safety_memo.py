import json
import warnings

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


def test_element_rust_json_has_the_same_checked_content_and_persisted_bytes(tmp_path):
    store = RunStore(tmp_path / "runs")
    checkpoint = checkpoint_with_element()
    request = store_module.AIRequestCheckpoint(
        stage="primary",
        model="модель 😀 / unicode \u2028 middle",
        cache_key="a" * 64,
        plan_sha256="b" * 64,
        request_sha256="c" * 64,
        shown_media_sha256=("d" * 64,),
        item_label="E001",
    )
    element = store_module.ElementCheckpoint(
        stage="ai_cached",
        ai_cache_key=request.cache_key,
        ai_requests=(request,),
        error_code='обычный текст: "\\\n😀',
    )
    checkpoint.elements["item"] = element
    assert json.loads(element.model_dump_json()) == element.model_dump(mode="json")
    expected = checkpoint.model_dump_json().encode("utf-8") + b"\n"
    assert store.save(checkpoint).read_bytes() == expected
    assert store.save(checkpoint).read_bytes() == expected
    assert store.load(checkpoint.run_id) == checkpoint


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_null_nonfinite_json_identity_never_hides_new_nested_secret(tmp_path, number):
    store = RunStore(tmp_path / "runs")
    checkpoint = checkpoint_with_element()
    nested = {"metric": None, "note": "ordinary"}
    checkpoint.elements["item"] = checkpoint.elements["item"].model_copy(
        update={"error_code": nested}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        path = store.save(checkpoint)
        # Rust writes nonfinite numbers as null, whereas mode=json retains the
        # float. Neither is secret text; a new key must still be checked afresh.
        nested["metric"] = number
        store.save(checkpoint)
        original = path.read_bytes()
        nested["api_key"] = "placeholder"
        with pytest.raises(RunStoreError, match="unsafe field"):
            store.save(checkpoint)
    assert path.read_bytes() == original


@pytest.mark.parametrize("shape", ["raw_element", "mapping_subclass", "model_subclass"])
def test_nonstandard_model_copy_containers_keep_full_safety_check(tmp_path, shape):
    store = RunStore(tmp_path / "runs")
    checkpoint = checkpoint_with_element()
    path = store.save(checkpoint)
    original = path.read_bytes()
    if shape == "raw_element":
        elements = {"item": {"stage": "discovered", "api_key": "placeholder"}}
    elif shape == "mapping_subclass":

        class Elements(dict):
            pass

        elements = Elements({"api_key": checkpoint.elements["item"]})
    else:

        class CustomElement(store_module.ElementCheckpoint):
            note: str

        elements = {
            "item": CustomElement(stage="discovered", note="ordinary").model_copy(
                update={"error_code": "https://user:password@example.test/"}
            )
        }
    unsafe = checkpoint.model_copy(update={"elements": elements})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(RunStoreError, match=r"unsafe field|credential"):
            store.save(unsafe)
    assert path.read_bytes() == original


def test_replaced_element_json_serializer_cannot_reuse_safe_payload(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "runs")
    checkpoint = checkpoint_with_element()
    safe = checkpoint.elements["item"].model_dump_json()
    path = store.save(checkpoint)
    original = path.read_bytes()
    checkpoint.elements["item"] = checkpoint.elements["item"].model_copy(
        update={"error_code": "https://user:password@example.test/"}
    )
    monkeypatch.setattr(store_module.ElementCheckpoint, "model_dump_json", lambda self: safe)
    with pytest.raises(RunStoreError, match="credential"):
        store.save(checkpoint)
    assert path.read_bytes() == original


@pytest.mark.parametrize("override", ["model_dump", "model_dump_json", "__pydantic_serializer__"])
def test_unvalidated_instance_serializer_override_cannot_hide_changed_content(tmp_path, override):
    store = RunStore(tmp_path / "runs")
    checkpoint = checkpoint_with_element()
    element = checkpoint.elements["item"]
    safe_json = element.model_dump_json()
    safe_payload = element.model_dump(mode="json")
    path = store.save(checkpoint)
    original = path.read_bytes()

    class SafeOnlySerializer:
        def to_json(self, *args, **kwargs):
            return safe_json.encode("utf-8")

        def to_python(self, *args, **kwargs):
            return safe_payload

    replacement = {
        "model_dump": lambda **kwargs: safe_payload,
        "model_dump_json": lambda **kwargs: safe_json,
        "__pydantic_serializer__": SafeOnlySerializer(),
    }[override]
    checkpoint.elements["item"] = element.model_copy(
        update={"error_code": "https://user:password@example.test/", override: replacement}
    )
    # The enclosing compiled writer ignores these unvalidated instance overrides.
    assert b"user:password" in checkpoint.model_dump_json().encode("utf-8")
    with pytest.raises(RunStoreError, match="credential"):
        store.save(checkpoint)
    assert path.read_bytes() == original


def test_element_memo_keeps_updated_budget_and_checks_renamed_nested_keys(tmp_path):
    store = RunStore(tmp_path / "runs")
    checkpoint = checkpoint_with_element()
    nested = {"note": "ordinary"}
    checkpoint.elements["item"] = checkpoint.elements["item"].model_copy(
        update={"error_code": nested}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        store.save(checkpoint)
        checkpoint = checkpoint.model_copy(update={"ai_requests_used": 7})
        path = store.save(checkpoint)
        assert json.loads(path.read_bytes())["ai_requests_used"] == 7
        original = path.read_bytes()
        nested["download_url"] = nested.pop("note")
        with pytest.raises(RunStoreError, match="unsafe field"):
            store.save(checkpoint)
    assert path.read_bytes() == original
