"""Tests for the harbor verifier corpus metric aggregation."""

from daydream.benchmark.harbor.verifier_core import aggregate_metrics


def _row(reward, tp, fp, fn, clean_task=0, clean_pass=0, verifier_error=0):
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) else 1.0
    return {"reward": reward, "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": f1, "gold_count": tp + fn, "candidate_count": tp + fp,
            "clean_task": clean_task, "clean_pass": clean_pass, "verifier_error": verifier_error}


def test_micro_metrics_pooled_not_averaged():
    rows = [_row(1.0, 3, 0, 0), _row(0.5, 1, 1, 1)]  # pooled tp=4 fp=1 fn=1
    m = aggregate_metrics(rows)
    assert abs(m["micro_precision"] - 0.8) < 1e-9  # 4/5
    assert abs(m["micro_recall"] - 0.8) < 1e-9  # 4/5
    assert abs(m["micro_f1"] - 0.8) < 1e-9
    assert m["micro_precision"] != 0.75  # averaging (1.0+0.5)/2 would give 0.75
    assert (m["total_tp"], m["total_fp"], m["total_fn"]) == (4, 1, 1)


def test_zero_denominator_metrics_are_one():
    m = aggregate_metrics([_row(0.0, 0, 0, 0)])
    assert (m["micro_precision"], m["micro_recall"], m["micro_f1"]) == (1.0, 1.0, 1.0)


def test_mean_task_score_and_counts():
    rows = [_row(1.0, 3, 0, 0), _row(0.5, 1, 1, 1), None, _row(0.0, 0, 0, 0, verifier_error=1)]
    m = aggregate_metrics(rows)
    assert m["task_count"] == 4
    assert m["failed_task_count"] == 2  # None + verifier_error
    assert abs(m["mean_task_score"] - (1.0 + 0.5 + 0.0 + 0.0) / 4) < 1e-9

def test_clean_accuracy_counts_only_clean_tasks():
    rows = [
        _row(1.0, 0, 0, 0, clean_task=1),  # correct clean (no FP)
        _row(0.0, 0, 2, 0, clean_task=1),  # failed clean (clean gold with FPs)
        _row(0.5, 1, 1, 1, clean_task=0),  # non-clean, excluded from denominator
    ]
    m = aggregate_metrics(rows)
    assert m["clean_task_count"] == 2
    assert abs(m["clean_accuracy"] - 0.5) < 1e-9  # 1 correct / 2 clean tasks


def test_clean_accuracy_zero_clean_tasks_is_one():
    m = aggregate_metrics([_row(0.5, 1, 1, 1, clean_task=0)])
    assert m["clean_task_count"] == 0
    assert m["clean_accuracy"] == 1.0  # zero-denominator → 1.0
