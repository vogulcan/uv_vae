"""Wall-clock timing for training runs.

These guard the thing the timer exists to prevent: a finished run whose duration
cannot be recovered afterwards. Every full-cohort run before this existed recorded
only decode seconds, which overlap the GPU and (with decode_workers > 1) are summed
across threads -- so they could not answer "how long did it take".
"""

from __future__ import annotations

import time

from uv_vae.training import RunTimer


def test_phases_sum_to_total() -> None:
    """setup + loop + finalize must reconstruct total, or the breakdown is a lie."""
    timer = RunTimer()
    time.sleep(0.02)
    timer.mark_setup_done()
    timer.epoch_start()
    time.sleep(0.02)
    timer.mark_loop_done()
    time.sleep(0.02)

    report = timer.as_dict()
    parts = (
        report["setup_seconds"]
        + report["train_loop_seconds"]
        + report["finalize_seconds"]
    )
    assert abs(report["total_seconds"] - parts) < 0.01
    assert report["total_seconds"] >= 0.06


def test_every_phase_is_non_negative() -> None:
    """perf_counter is monotonic, so no phase can come out negative."""
    timer = RunTimer()
    timer.mark_setup_done()
    timer.mark_loop_done()
    report = timer.as_dict()
    for key in ("total_seconds", "setup_seconds", "train_loop_seconds", "finalize_seconds"):
        assert report[key] >= 0.0, f"{key} went negative"


def test_epoch_seconds_measures_between_marks() -> None:
    timer = RunTimer()
    timer.epoch_start()
    time.sleep(0.03)
    first = timer.epoch_seconds()

    timer.epoch_start()
    time.sleep(0.01)
    second = timer.epoch_seconds()

    assert first >= 0.03
    assert second < first, "epoch_start must reset the clock, not accumulate"


def test_epoch_seconds_without_start_is_zero_not_a_crash() -> None:
    """A trainer that never calls epoch_start still has to write its report."""
    assert RunTimer().epoch_seconds() == 0.0


def test_unfinished_phases_are_null_not_zero() -> None:
    """A killed run must not report 0s of training -- null says 'never recorded'."""
    report = RunTimer().as_dict()
    assert report["setup_seconds"] is None
    assert report["train_loop_seconds"] is None
    assert report["finalize_seconds"] is None
    assert report["total_seconds"] >= 0.0, "total is always measurable"


def test_epoch_aggregates_travel_with_the_report() -> None:
    """train_result.json keeps only final_epoch, so the aggregates must live here."""
    timer = RunTimer()
    for _ in range(3):
        timer.epoch_start()
        time.sleep(0.01)
        timer.epoch_seconds()

    report = timer.as_dict()
    assert report["epochs_timed"] == 3
    assert (
        report["epoch_min_seconds"]
        <= report["epoch_mean_seconds"]
        <= report["epoch_max_seconds"]
    )


def test_zero_epochs_does_not_divide_by_zero() -> None:
    """A run that dies during setup still has to write its report."""
    report = RunTimer().as_dict()
    assert report["epochs_timed"] == 0
    assert report["epoch_mean_seconds"] is None
    assert report["epoch_min_seconds"] is None
    assert report["epoch_max_seconds"] is None


def test_hours_matches_seconds() -> None:
    timer = RunTimer()
    timer.mark_setup_done()
    timer.mark_loop_done()
    report = timer.as_dict()
    assert abs(report["total_hours"] - report["total_seconds"] / 3600.0) < 1e-6


def test_timestamps_are_iso_and_ordered() -> None:
    from datetime import datetime

    timer = RunTimer()
    time.sleep(0.01)
    report = timer.as_dict()
    started = datetime.fromisoformat(report["started_utc"])
    finished = datetime.fromisoformat(report["finished_utc"])
    assert finished >= started
