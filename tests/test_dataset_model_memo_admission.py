"""Bounded whole-dataset scans retain reusable exact-byte parsed models."""

import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Barrier

import pytest
from pydantic import BaseModel

from mojilex_cli.dataset import repository


class CacheRecord(BaseModel):
    number: int
    labels: list[str]


def record_bytes(number: int, label: str = "one") -> bytes:
    return json.dumps({"number": number, "labels": [label]}).encode()


@pytest.mark.parametrize("limiting_cap", ["entries", "bytes"])
def test_full_scans_keep_admitted_records_without_copying_the_uncached_tail(
    monkeypatch, limiting_cap
):
    records = [record_bytes(number) for number in range(10)]
    assert len(set(map(len, records))) == 1
    monkeypatch.setattr(repository, "_MODEL_CACHE_ENTRIES", 4 if limiting_cap == "entries" else 20)
    monkeypatch.setattr(
        repository,
        "_MODEL_CACHE_BYTES",
        sum(map(len, records[:4])) if limiting_cap == "bytes" else 4096,
    )
    validations = []
    copies = []
    real_validate = CacheRecord.model_validate
    real_copy = CacheRecord.model_copy

    def validate(cls, value, *args, **kwargs):
        validations.append(value["number"])
        return real_validate(value, *args, **kwargs)

    def copied(self, *args, **kwargs):
        copies.append(self.number)
        return real_copy(self, *args, **kwargs)

    monkeypatch.setattr(CacheRecord, "model_validate", classmethod(validate))
    monkeypatch.setattr(CacheRecord, "model_copy", copied)
    with repository.dataset_read_scope():
        for pass_number in range(3):
            found = [
                repository._read_models(data, CacheRecord, source="record", many=False)[0]
                for data in records
            ]
            assert [item.number for item in found] == list(range(10))
            assert all(item.labels == ["one"] for item in found)
            # Both admitted and uncached values must belong to this caller.
            found[0].labels.append("mutated")
            found[-1].labels.append("mutated")
            memo = repository._MODEL_MEMO.get()
            assert len(memo.entries) == 4
            assert memo.size == sum(map(len, records[:4]))
            assert len(validations) == 10 + pass_number * 6
            assert len(copies) == (pass_number + 1) * 4
        assert set(copies) == {0, 1, 2, 3}


def test_changed_equal_length_bytes_validate_fresh_when_admission_is_full(monkeypatch):
    monkeypatch.setattr(repository, "_MODEL_CACHE_ENTRIES", 1)
    validations = []
    real_validate = CacheRecord.model_validate

    def validate(cls, value, *args, **kwargs):
        validations.append(value)
        return real_validate(value, *args, **kwargs)

    monkeypatch.setattr(CacheRecord, "model_validate", classmethod(validate))
    original = record_bytes(0, "one")
    changed = record_bytes(0, "two")
    assert len(original) == len(changed)
    with repository.dataset_read_scope():
        first = repository._read_models(original, CacheRecord, source="first", many=False)[0]
        first.labels.append("caller mutation")
        revised = repository._read_models(changed, CacheRecord, source="changed", many=False)[0]
        assert revised.labels == ["two"]
        revised.labels.append("caller mutation")
        revised_again = repository._read_models(changed, CacheRecord, source="changed", many=False)[
            0
        ]
        assert revised_again.labels == ["two"]
        retained = repository._read_models(original, CacheRecord, source="retained", many=False)[0]
        assert retained.labels == ["one"]
        assert len(validations) == 3
        assert len(repository._MODEL_MEMO.get().entries) == 1
    assert repository._MODEL_MEMO.get() is None


def test_concurrent_misses_cannot_exceed_admission_limits(monkeypatch):
    records = [record_bytes(number) for number in range(4)]
    monkeypatch.setattr(repository, "_MODEL_CACHE_ENTRIES", 2)
    monkeypatch.setattr(repository, "_MODEL_CACHE_BYTES", len(records[0]) * 2)
    gate = Barrier(4)
    real_validate = CacheRecord.model_validate

    def validate(cls, value, *args, **kwargs):
        result = real_validate(value, *args, **kwargs)
        gate.wait(timeout=10)
        return result

    monkeypatch.setattr(CacheRecord, "model_validate", classmethod(validate))
    with repository.dataset_read_scope(), ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                copy_context().run,
                repository._read_models,
                data,
                CacheRecord,
                source="concurrent record",
                many=False,
            )
            for data in records
        ]
        results = [future.result(timeout=15)[0] for future in futures]
        assert [item.number for item in results] == list(range(4))
        memo = repository._MODEL_MEMO.get()
        assert len(memo.entries) == 2
        assert memo.size == len(records[0]) * 2
