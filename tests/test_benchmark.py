"""The latency benchmark's bookkeeping, checked with a fake model and a fake clock-free workload."""

import numpy as np
import pytest

import benchmark_latency as bl


def test_percentiles_match_hand_computed_values():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert bl.percentile(values, 50) == pytest.approx(5.5)
    assert bl.percentile(values, 95) == pytest.approx(9.55)


def test_summarize_converts_seconds_to_ms_and_computes_throughput():
    row = bl.summarize(8, [0.010, 0.020, 0.030, 0.040])
    assert row["batch_p50_ms"] == pytest.approx(25.0)
    assert row["batch_p95_ms"] == pytest.approx(38.5)
    assert row["per_image_p50_ms"] == pytest.approx(25.0 / 8)
    # 4 iterations x 8 images over 0.100 s of total timed work
    assert row["throughput_img_per_s"] == pytest.approx(320.0)


def test_warmup_calls_are_made_but_excluded_from_the_timings():
    calls = []

    def predict(batch):
        calls.append(len(batch))

    rows = bl.benchmark(predict, bl.random_batch_factory(), batch_sizes=(1, 4), warmup=3, iterations=5)
    assert calls == [1] * 8 + [4] * 8          # 3 warm-up + 5 timed, per batch size
    assert [r["batch_size"] for r in rows] == [1, 4]
    assert all(r["iterations"] == 5 for r in rows)  # warm-up never counted


def test_warmup_calls_read_no_clock_so_only_timed_calls_are_measured(monkeypatch):
    clock = iter(
        # warm-up calls consume no clock reads; each timed call reads the clock twice
        [0.0, 0.010, 0.010, 0.020, 0.020, 0.030]
    )
    monkeypatch.setattr(bl.time, "perf_counter", lambda: next(clock))
    state = {"n": 0}

    def predict(batch):
        state["n"] += 1

    row = bl.benchmark(predict, lambda n: np.zeros((n, 2)), batch_sizes=(1,), warmup=2, iterations=3)[0]
    assert state["n"] == 5
    assert row["batch_p50_ms"] == pytest.approx(10.0) and row["batch_p95_ms"] == pytest.approx(10.0)


def test_bad_arguments_are_rejected():
    with pytest.raises(ValueError):
        bl.benchmark(lambda b: None, lambda n: np.zeros(n), iterations=0)


def test_random_batches_have_the_model_input_shape_and_range():
    batch = bl.random_batch_factory((224, 224))(3)
    assert batch.shape == (3, 224, 224, 3) and batch.dtype == np.float32
    assert 0 <= batch.min() and batch.max() <= 255


def test_main_requires_a_model_or_an_http_target(capsys):
    with pytest.raises(SystemExit):
        bl.main([])
    with pytest.raises(SystemExit):
        bl.main(["--http", "http://localhost:1/predict"])
