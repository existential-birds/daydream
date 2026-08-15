import json

import pytest

from daydream.benchmark.anthropic_score import (
    AnthropicJsonClient,
    run_anthropic_dedup,
    run_anthropic_extraction,
    run_anthropic_scoring,
)
from daydream.benchmark.score import model_results_dir

URL = "https://x/pull/1"
UNSELECTED_URL = "https://x/pull/2"
UNSELECTED_SENTINEL = "unselected review must not be scored"


class FakeAnthropicJson:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    async def complete_json(self, *, system, user, max_tokens):
        self.requests.append((system, user, max_tokens))
        return self._responses.pop(0)


def seed_benchmark_data(tmp_path, *, tool, body):
    results = tmp_path / "results"
    results.mkdir()
    (results / "benchmark_data.json").write_text(
        json.dumps(
            {
                URL: {
                    "golden_comments": [{"comment": body, "severity": "medium"}],
                    "reviews": [
                        {
                            "tool": tool,
                            "repo_name": "repo",
                            "pr_url": URL,
                            "review_comments": [{"body": body}],
                        }
                    ]
                }
            }
        )
    )


def seed_candidates(tmp_path, *, model, tool, texts):
    scores_dir = model_results_dir(tmp_path, model)
    scores_dir.mkdir(parents=True)
    (scores_dir / "candidates.json").write_text(
        json.dumps({URL: {tool: [{"text": text, "path": None, "line": None} for text in texts]}})
    )


def seed_dedup_groups(tmp_path, *, model, tool, groups):
    scores_dir = model_results_dir(tmp_path, model)
    scores_dir.mkdir(parents=True, exist_ok=True)
    (scores_dir / "dedup_groups.json").write_text(json.dumps({URL: {tool: groups}}))


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_anthropic_json_client_posts_messages_and_parses_json():
    calls = []

    class FakeClient:
        async def post(self, url, *, headers, json, timeout):
            calls.append((url, headers, json, timeout))
            return FakeResponse(200, {"content": [{"type": "text", "text": '{"issues":["x"]}'}]})

    client = AnthropicJsonClient(api_key="sk-ant-x", model="claude-opus-4-5-20251101", http=FakeClient())
    result = await client.complete_json(system="extract", user="return json", max_tokens=128)
    assert result == {"issues": ["x"]}
    assert calls[0][0] == "https://api.anthropic.com/v1/messages"
    assert calls[0][1]["x-api-key"] == "sk-ant-x"
    assert calls[0][2]["model"] == "claude-opus-4-5-20251101"
    assert calls[0][2]["temperature"] == 0


@pytest.mark.asyncio
async def test_direct_extraction_writes_martian_candidates(tmp_path):
    seed_benchmark_data(tmp_path, tool="daydream", body="Bug one.\n\nBug two in cache keys.")
    client = FakeAnthropicJson([{"issues": ["Bug one", "Bug two"]}])

    scores_dir = model_results_dir(tmp_path, "claude-opus-4-5-20251101")
    await run_anthropic_extraction(tmp_path, "claude-opus-4-5-20251101", tool="daydream", client=client)

    candidates = json.loads((scores_dir / "candidates.json").read_text())
    leaf = candidates[URL]["daydream"]
    assert [c["text"] for c in leaf] == ["Bug one", "Bug two"]
    assert all(c["source"] == "extracted" and c["path"] is None and c["line"] is None for c in leaf)


@pytest.mark.parametrize(
    ("texts", "response", "expected"),
    [
        (["same bug", "same issue"], {"groups": [[0, 1]]}, [[0, 1]]),
        (["a", "b"], {"groups": [[0, 0]]}, [[0], [1]]),
    ],
    ids=["valid-groups", "invalid-groups-singletons"],
)
@pytest.mark.asyncio
async def test_direct_dedup_writes_groups_or_singletons(tmp_path, texts, response, expected):
    """Persist model-supplied dedup groups or fallback singleton groups."""
    seed_candidates(tmp_path, model="claude-opus-4-5-20251101", tool="daydream", texts=texts)
    client = FakeAnthropicJson([response])

    await run_anthropic_dedup(tmp_path, "claude-opus-4-5-20251101", tool="daydream", client=client)

    groups = json.loads((model_results_dir(tmp_path, "claude-opus-4-5-20251101") / "dedup_groups.json").read_text())
    assert groups[URL]["daydream"] == expected


@pytest.mark.asyncio
async def test_direct_judge_writes_evaluations_and_metadata(tmp_path):
    seed_benchmark_data(tmp_path, tool="daydream", body="candidate")
    seed_candidates(tmp_path, model="claude-opus-4-5-20251101", tool="daydream", texts=["candidate"])
    seed_dedup_groups(tmp_path, model="claude-opus-4-5-20251101", tool="daydream", groups=[[0]])
    client = FakeAnthropicJson(
        [
            {"issues": ["candidate"]},
            {"reasoning": "same bug", "match": True, "confidence": 0.91},
        ]
    )

    scores = await run_anthropic_scoring(
        tmp_path,
        "claude-opus-4-5-20251101",
        golden_urls=[URL],
        tool="daydream",
        client=client,
    )

    assert scores.scored_pr_count == 1
    assert scores.total_tp == 1 and scores.total_fp == 0 and scores.total_fn == 0
    evals = json.loads((model_results_dir(tmp_path, "claude-opus-4-5-20251101") / "evaluations.json").read_text())
    leaf = evals[URL]["daydream"]
    assert leaf["judge_route"] == "anthropic-direct"
    assert leaf["judge_model"] == "claude-opus-4-5-20251101"
    assert leaf["tp"] == 1 and leaf["precision"] == 1.0 and leaf["recall"] == 1.0


@pytest.mark.asyncio
async def test_direct_judge_does_not_reuse_one_candidate_for_two_goldens(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "benchmark_data.json").write_text(
        json.dumps(
            {
                URL: {
                    "golden_comments": [
                        {"comment": "golden one", "severity": "medium"},
                        {"comment": "golden two", "severity": "medium"},
                    ],
                    "reviews": [
                        {
                            "tool": "daydream",
                            "repo_name": "repo",
                            "pr_url": URL,
                            "review_comments": [{"body": "candidate"}],
                        }
                    ],
                }
            }
        )
    )
    seed_candidates(tmp_path, model="claude-opus-4-5-20251101", tool="daydream", texts=["candidate"])
    seed_dedup_groups(tmp_path, model="claude-opus-4-5-20251101", tool="daydream", groups=[[0]])
    client = FakeAnthropicJson(
        [
            {"issues": ["candidate"]},
            {"reasoning": "matches one", "match": True, "confidence": 0.7},
            {"reasoning": "matches two", "match": True, "confidence": 0.9},
        ]
    )

    scores = await run_anthropic_scoring(
        tmp_path,
        "claude-opus-4-5-20251101",
        golden_urls=[URL],
        tool="daydream",
        client=client,
    )

    assert scores.total_tp == 1 and scores.total_fn == 1
    leaf = json.loads(
        (model_results_dir(tmp_path, "claude-opus-4-5-20251101") / "evaluations.json").read_text()
    )[URL]["daydream"]
    assert leaf["tp"] == 1
    assert leaf["precision"] <= 1.0
    assert [tp["golden_comment"] for tp in leaf["true_positives"]] == ["golden two"]
    assert [fn["golden_comment"] for fn in leaf["false_negatives"]] == ["golden one"]


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            {
                "name": "duplicate-goldens",
                "golden_comments": [
                    {"comment": "same bug", "severity": "medium"},
                    {"comment": "same bug", "severity": "medium"},
                ],
                "issues": ["same bug"],
                "groups": [[0]],
                "verdicts": [
                    {"reasoning": "g0", "match": True, "confidence": 0.9},
                    {"reasoning": "g1", "match": True, "confidence": 0.8},
                ],
                "expected_tp": [
                    {"golden_comment": "same bug", "severity": "medium",
                     "matched_candidate": "same bug", "confidence": 0.9,
                     "reasoning": "g0"},
                ],
                "expected_fp": [],
                "expected_fn": [{"golden_comment": "same bug", "severity": "medium"}],
                "tp": 1, "fp": 0, "fn": 1,
                "precision": 1.0, "recall": 0.5,
                "total_tp": 1, "total_fp": 0, "total_fn": 1,
            },
            id="duplicate-goldens",
        ),
        pytest.param(
            {
                "name": "duplicate-candidates",
                "golden_comments": [{"comment": "the bug", "severity": "medium"}],
                "issues": ["the bug", "the bug"],
                "groups": [[0], [1]],
                "verdicts": [
                    {"reasoning": "c0", "match": True, "confidence": 0.9},
                    {"reasoning": "c1", "match": True, "confidence": 0.8},
                ],
                "expected_tp": [
                    {"golden_comment": "the bug", "severity": "medium",
                     "matched_candidate": "the bug", "confidence": 0.9,
                     "reasoning": "c0"},
                ],
                "expected_fp": [{"candidate": "the bug"}],
                "expected_fn": [],
                "tp": 1, "fp": 1, "fn": 0,
                "precision": 0.5, "recall": 1.0,
                "total_tp": 1, "total_fp": 1, "total_fn": 0,
            },
            id="duplicate-candidates",
        ),
    ],
)
@pytest.mark.asyncio
async def test_direct_judge_tracks_duplicate_text_occurrences_by_position(tmp_path, case):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "benchmark_data.json").write_text(json.dumps({
        URL: {
            "golden_comments": case["golden_comments"],
            "reviews": [{"tool": "daydream", "repo_name": "repo", "pr_url": URL,
                         "review_comments": [{"body": "the bug"}]}],
        }
    }))
    seed_candidates(tmp_path, model="claude-opus-4-5-20251101", tool="daydream", texts=case["issues"])
    seed_dedup_groups(tmp_path, model="claude-opus-4-5-20251101", tool="daydream", groups=case["groups"])
    # Dedup only calls the completer when there are >= 2 candidates (_MIN_DEDUP_CANDIDATES);
    # with a single candidate no groups request/response is emitted, so the queue
    # must mirror the actual call order to keep verdicts aligned with judge tasks.
    client_queue = [{"issues": case["issues"]}]
    if len(case["issues"]) >= 2:
        client_queue.append({"groups": case["groups"]})
    client_queue += case["verdicts"]
    client = FakeAnthropicJson(client_queue)

    scores = await run_anthropic_scoring(
        tmp_path, "claude-opus-4-5-20251101", golden_urls=[URL], tool="daydream", client=client
    )

    leaf = json.loads(
        (model_results_dir(tmp_path, "claude-opus-4-5-20251101") / "evaluations.json").read_text()
    )[URL]["daydream"]
    assert leaf["true_positives"] == case["expected_tp"]
    assert leaf["false_positives"] == case["expected_fp"]
    assert leaf["false_negatives"] == case["expected_fn"]
    assert (leaf["tp"], leaf["fp"], leaf["fn"]) == (case["tp"], case["fp"], case["fn"])
    assert leaf["precision"] == pytest.approx(case["precision"])
    assert leaf["recall"] == pytest.approx(case["recall"])
    assert all(
        "golden_index" not in rec and "candidate_index" not in rec
        for rec in leaf["true_positives"] + leaf["false_positives"] + leaf["false_negatives"]
    )
    assert (scores.total_tp, scores.total_fp, scores.total_fn) == (
        case["total_tp"], case["total_fp"], case["total_fn"]
    )


@pytest.mark.asyncio
async def test_direct_judge_precision_counts_dedup_siblings_once(tmp_path):
    """A dedup sibling group is one logical candidate: matching it must not leave
    its siblings inflating the precision denominator behind ``fp``'s back."""
    seed_benchmark_data(tmp_path, tool="daydream", body="same bug, said three ways")
    client = FakeAnthropicJson(
        [
            {"issues": ["same bug A", "same bug B", "same bug C"]},
            {"groups": [[0, 1, 2]]},
            {"reasoning": "same", "match": True, "confidence": 0.9},
            {"reasoning": "same", "match": True, "confidence": 0.8},
            {"reasoning": "same", "match": True, "confidence": 0.7},
        ]
    )

    scores = await run_anthropic_scoring(
        tmp_path,
        "claude-opus-4-5-20251101",
        golden_urls=[URL],
        tool="daydream",
        client=client,
    )

    leaf = json.loads(
        (model_results_dir(tmp_path, "claude-opus-4-5-20251101") / "evaluations.json").read_text()
    )[URL]["daydream"]
    assert leaf["tp"] == 1 and leaf["fp"] == 0
    assert leaf["total_candidates"] == 3
    assert leaf["precision"] == 1.0
    assert leaf["precision"] == leaf["tp"] / (leaf["tp"] + leaf["fp"])
    assert scores.precision == 1.0


@pytest.mark.asyncio
async def test_direct_route_artifacts_are_martian_compatible(tmp_path):
    seed_benchmark_data(tmp_path, tool="daydream", body="candidate")
    client = FakeAnthropicJson(
        [
            {"issues": ["candidate"]},
            {"reasoning": "same", "match": True, "confidence": 1.0},
        ]
    )
    scores = await run_anthropic_scoring(
        tmp_path,
        "claude-opus-4-5-20251101",
        golden_urls=[URL],
        tool="daydream",
        client=client,
    )
    model_dir = model_results_dir(tmp_path, "claude-opus-4-5-20251101")
    assert (model_dir / "candidates.json").exists()
    assert (model_dir / "dedup_groups.json").exists()
    assert (model_dir / "evaluations.json").exists()
    assert scores.total_tp == 1


@pytest.mark.asyncio
async def test_direct_route_recomputes_existing_artifacts_for_current_review(tmp_path):
    seed_benchmark_data(tmp_path, tool="daydream", body="new candidate")
    model = "claude-opus-4-5-20251101"
    model_dir = model_results_dir(tmp_path, model)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "candidates.json").write_text(
        json.dumps({URL: {"daydream": [{"text": "stale candidate", "path": None, "line": None}]}})
    )
    (model_dir / "evaluations.json").write_text(
        json.dumps({URL: {"daydream": {"errors_count": 0, "tp": 1, "fp": 0, "fn": 0}}})
    )
    client = FakeAnthropicJson(
        [
            {"issues": ["new candidate"]},
            {"reasoning": "same", "match": True, "confidence": 1.0},
        ]
    )

    scores = await run_anthropic_scoring(tmp_path, model, golden_urls=[URL], tool="daydream", client=client)

    candidates = json.loads((model_dir / "candidates.json").read_text())
    evals = json.loads((model_dir / "evaluations.json").read_text())
    assert [c["text"] for c in candidates[URL]["daydream"]] == ["new candidate"]
    assert evals[URL]["daydream"]["total_candidates"] == 1
    assert scores.total_tp == 1


@pytest.mark.asyncio
async def test_direct_scoring_filters_extraction_and_dedup_to_selected_urls(tmp_path):
    model = "claude-opus-4-5-20251101"
    model_dir = model_results_dir(tmp_path, model)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Two benchmark-data entries: selected (URL) and unselected (UNSELECTED_URL,
    # whose golden comment + review body carry the sentinel).
    (tmp_path / "results" / "benchmark_data.json").write_text(json.dumps({
        URL: {
            "golden_comments": [{"comment": "Bug one", "severity": "medium"}],
            "reviews": [{"tool": "daydream", "repo_name": "repo", "pr_url": URL,
                         "review_comments": [{"body": "Bug one.\n\nBug two in cache keys."}]}],
        },
        UNSELECTED_URL: {
            "golden_comments": [{"comment": UNSELECTED_SENTINEL, "severity": "medium"}],
            "reviews": [{"tool": "daydream", "repo_name": "repo", "pr_url": UNSELECTED_URL,
                         "review_comments": [{"body": UNSELECTED_SENTINEL}]}],
        },
    }))

    # Stale selected leaves + preserved unselected leaves (sentinel-bearing).
    preserved_candidates = [{"text": UNSELECTED_SENTINEL, "path": None, "line": None},
                               {"text": UNSELECTED_SENTINEL, "path": None, "line": None}]
    (model_dir / "candidates.json").write_text(json.dumps({
        URL: {"daydream": [{"text": "stale selected candidate", "path": None, "line": None}]},
        UNSELECTED_URL: {"daydream": preserved_candidates},
    }))
    preserved_groups = [[0], [1]]
    (model_dir / "dedup_groups.json").write_text(json.dumps({
        URL: {"daydream": [[0]]},
        UNSELECTED_URL: {"daydream": preserved_groups},
    }))

    client = FakeAnthropicJson([
        {"issues": ["Bug one", "Bug two"]},          # extraction (selected only)
        {"groups": [[0, 1]]},                         # dedup (selected only)
        {"reasoning": "matches", "match": True, "confidence": 0.9},    # judge: golden x cand 0
        {"reasoning": "no match", "match": False, "confidence": 0.2},  # judge: golden x cand 1
    ])

    scores = await run_anthropic_scoring(
        tmp_path, model, golden_urls=[URL], tool="daydream", client=client,
    )

    # Exactly four completions: extraction, dedup, two golden-candidate verdicts.
    assert len(client.requests) == 4
    # No model work touches unselected PR text.
    assert all(UNSELECTED_SENTINEL not in req[1] for req in client.requests)

    # Selected leaves replaced with fresh artifacts.
    candidates = json.loads((model_dir / "candidates.json").read_text())
    assert [c["text"] for c in candidates[URL]["daydream"]] == ["Bug one", "Bug two"]
    dedup = json.loads((model_dir / "dedup_groups.json").read_text())
    assert dedup[URL]["daydream"] == [[0, 1]]

    # Unselected leaves structurally unchanged (load-then-merge preservation).
    assert candidates[UNSELECTED_URL]["daydream"] == preserved_candidates
    assert dedup[UNSELECTED_URL]["daydream"] == preserved_groups

    # Selected-only scoring.
    assert scores.scored_pr_count == 1
    assert scores.total_tp == 1
