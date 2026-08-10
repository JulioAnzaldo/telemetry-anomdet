"""Event-level scoring must match telemanom's, including its asymmetries."""

import pytest

from telemetry_anomdet.evaluation import evaluate_sequences, sequence_prf


def test_exact_match():
    got = evaluate_sequences([(10, 20)], [(10, 20)])
    assert (got["true_positives"], got["false_positives"], got["false_negatives"]) == (1, 0, 0)


def test_partial_overlap_counts_as_caught():
    """Touching a labelled event anywhere is a catch; extent is not scored."""
    got = evaluate_sequences([(18, 25)], [(10, 20)])
    assert got["true_positives"] == 1
    assert got["false_positives"] == 0


def test_prediction_touching_nothing_is_a_false_positive():
    got = evaluate_sequences([(50, 60)], [(10, 20)])
    assert (got["true_positives"], got["false_positives"], got["false_negatives"]) == (0, 1, 1)
    assert got["fp_sequences"] == [(50, 60)]


def test_one_prediction_spanning_two_events_credits_only_the_first():
    """
    A single alarm covering two labelled events is one catch, not two.

    This is the asymmetry that makes the metric conservative, and it is easy to
    get wrong by asking 'was this event touched' per event instead.
    """
    got = evaluate_sequences([(5, 100)], [(10, 20), (60, 70)])
    assert got["true_positives"] == 1
    assert got["false_negatives"] == 1
    assert got["false_positives"] == 0


def test_two_predictions_on_one_event_are_not_double_counted():
    got = evaluate_sequences([(10, 12), (15, 18)], [(10, 20)])
    assert got["true_positives"] == 1
    assert got["false_positives"] == 0
    assert len(got["tp_sequences"]) == 2


def test_true_positives_and_false_negatives_partition_the_labels():
    got = evaluate_sequences([(10, 20)], [(10, 20), (60, 70), (90, 95)])
    assert got["true_positives"] + got["false_negatives"] == 3


def test_no_predictions_gives_all_false_negatives():
    got = evaluate_sequences([], [(10, 20), (60, 70)])
    assert (got["true_positives"], got["false_positives"], got["false_negatives"]) == (0, 0, 2)


def test_no_labels_makes_every_prediction_false():
    got = evaluate_sequences([(1, 2), (5, 6)], [])
    assert (got["true_positives"], got["false_positives"], got["false_negatives"]) == (0, 2, 0)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_pools_counts_before_dividing():
    rows = [
        {"true_positives": 3, "false_positives": 1, "false_negatives": 0},
        {"true_positives": 0, "false_positives": 0, "false_negatives": 2},
    ]
    got = sequence_prf(rows)
    assert got["precision"] == pytest.approx(3 / 4)
    assert got["recall"] == pytest.approx(3 / 5)


def test_aggregate_reproduces_the_published_smap_totals():
    """telemanom reports TP 46, FP 10, FN 5 for SMAP; the maths must agree."""
    got = sequence_prf([{"true_positives": 46, "false_positives": 10, "false_negatives": 5}])
    assert got["precision"] == pytest.approx(0.821, abs=0.001)
    assert got["recall"] == pytest.approx(0.902, abs=0.001)
    assert got["f1"] == pytest.approx(0.859, abs=0.001)


def test_aggregate_of_nothing_is_zero_not_an_error():
    got = sequence_prf([])
    assert got["precision"] == 0.0 and got["recall"] == 0.0 and got["f1"] == 0.0
