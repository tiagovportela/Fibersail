"""Tests for the upload transport doubles (Part 3)."""

from __future__ import annotations

import random

import pytest

from fibersail_edge.cloud import (
    FlakyUploader,
    InMemoryUploader,
    TransientUploadError,
    Uploader,
)


def test_doubles_conform_to_protocol() -> None:
    assert isinstance(InMemoryUploader(), Uploader)
    assert isinstance(FlakyUploader(InMemoryUploader()), Uploader)


def test_inmemory_stores_by_key() -> None:
    up = InMemoryUploader()
    up.upload("a/b.gz", b"one")
    up.upload("a/b.gz", b"two")  # same key overwrites (idempotent)
    up.upload("c/d.gz", b"three")
    assert up.objects == {"a/b.gz": b"two", "c/d.gz": b"three"}


def test_flaky_fail_first_n_then_succeeds() -> None:
    inner = InMemoryUploader()
    flaky = FlakyUploader(inner, fail_first_n=3)
    for _ in range(3):
        with pytest.raises(TransientUploadError):
            flaky.upload("k", b"v")
    flaky.upload("k", b"v")  # 4th attempt succeeds
    assert inner.objects == {"k": b"v"}
    assert flaky.attempts == 4


def test_flaky_pattern_cycles() -> None:
    inner = InMemoryUploader()
    flaky = FlakyUploader(inner, pattern=[True, False])  # fail, ok, fail, ok, ...
    with pytest.raises(TransientUploadError):
        flaky.upload("k", b"1")
    flaky.upload("k", b"2")
    with pytest.raises(TransientUploadError):
        flaky.upload("k", b"3")
    assert inner.objects == {"k": b"2"}


def test_flaky_probabilistic_is_seed_reproducible() -> None:
    def run() -> int:
        flaky = FlakyUploader(InMemoryUploader(), fail_prob=0.5, rng=random.Random(123))
        failures = 0
        for _ in range(200):
            try:
                flaky.upload("k", b"v")
            except TransientUploadError:
                failures += 1
        return failures

    assert run() == run()  # same seed → same failure sequence


def test_flaky_validation() -> None:
    with pytest.raises(ValueError):
        FlakyUploader(InMemoryUploader(), fail_prob=1.5)
    with pytest.raises(ValueError):
        FlakyUploader(InMemoryUploader(), fail_first_n=-1)
    with pytest.raises(ValueError):
        FlakyUploader(InMemoryUploader(), pattern=[])
