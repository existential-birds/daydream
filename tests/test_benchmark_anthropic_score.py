import asyncio
import json

import pytest

from daydream.benchmark.anthropic_score import (
    _JUDGE_CONCURRENCY,
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


class OutOfOrderAnthropicJson:
    """Completes later-scheduled requests first; tracks concurrent overlap."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._next_index = 0
        self.active = 0
        self.peak_active = 0

    async def complete_json(self, *, system, user, max_tokens):
        index = self._next_index
        self._next_index += 1
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            # Earlier indices yield more event-loop turns, so they finish last.
            for _ in range(len(self._responses) - index):
                await asyncio.sleep(0)
            response = self._responses[index]
            if isinstance(response, Exception):
                raise response
            return response
        finally:
            self.active -= 1


def seed_benchmark_data(tmp_path, *, tool, body):
    seed_parallel_benchmark_data(tmp_path, tool=tool, reviews_by_url={URL: [body]})


def seed_parallel_benchmark_data(tmp_path, *, tool, reviews_by_url):
    """Write results/benchmark_data.json from an ordered {url: [bodies]} map."""
    (tmp_path / "results").mkdir()
    data = {}
    for golden_url, bodies in reviews_by_url.items():
        data[golden_url] = {
            "golden_comments": [{"comment": bodies[-1], "severity": "medium"}],
            "reviews": [
                {
                    "tool": tool,
                    "repo_name": "repo",
                    "pr_url": golden_url,
                    "review_comments": [{"body": body}],
                }
                for body in bodies
            ],
        }
    (tmp_path / "results" / "benchmark_data.json").write_text(json.dumps(data))


def seed_candidates(tmp_path, *, model, tool, texts):
    seed_parallel_candidates(tmp_path, model=model, tool=tool, texts_by_url={URL: texts})


def seed_parallel_candidates(tmp_path, *, model, tool, texts_by_url):
    """Write an ordered multi-URL candidates.json using the current field shape."""
    scores_dir = model_results_dir(tmp_path, model)
    scores_dir.mkdir(parents=True)
    (scores_dir / "candidates.json").write_text(json.dumps({
        url: {tool: [{"text": text, "path": None, "line": None} for text in texts]}
        for url, texts in texts_by_url.items()
    }))


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


@pytest.mark.asyncio
async def test_direct_extraction_bounds_concurrency_and_preserves_source_order(tmp_path):
    urls = [f"https://x/pull/{i}" for i in range(1, 12)]  # 11 URLs
    reviews_by_url = {urls[0]: ["first review", "replacement review"]}
    for url in urls[1:]:
        reviews_by_url[url] = [f"review for {url}"]
    seed_parallel_benchmark_data(tmp_path, tool="daydream", reviews_by_url=reviews_by_url)

    responses = [{"issues": ["first issue"]}, {"issues": ["replacement issue"]}]
    responses += [{"issues": [f"issue for {url}"]} for url in urls[1:]]
    client = OutOfOrderAnthropicJson(responses)  # 12 requests

    await run_anthropic_extraction(tmp_path, "claude-opus-4-5-20251101", tool="daydream", client=client)

    assert client.peak_active == _JUDGE_CONCURRENCY, f"peak={client.peak_active}"
    candidates = json.loads(
        (model_results_dir(tmp_path, "claude-opus-4-5-20251101") / "candidates.json").read_text()
    )
    assert list(candidates.keys()) == urls
    assert [c["text"] for c in candidates[urls[0]]["daydream"]] == ["replacement issue"]
    for url in urls[1:]:
        assert [c["text"] for c in candidates[url]["daydream"]] == [f"issue for {url}"]


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
async def test_direct_dedup_bounds_concurrency_preserves_order_and_falls_back_per_entry(tmp_path):
    urls = [f"https://x/pull/{i}" for i in range(1, 13)]  # 12 URLs
    texts_by_url = {url: [f"text a for {url}", f"text b for {url}"] for url in urls}
    seed_parallel_candidates(tmp_path, model="claude-opus-4-5-20251101", tool="daydream",
                             texts_by_url=texts_by_url)

    responses = [{"groups": [[0, 1]]} for _ in urls]
    responses[4] = RuntimeError("dedup request failed")  # 5th entry fails
    client = OutOfOrderAnthropicJson(responses)

    await run_anthropic_dedup(tmp_path, "claude-opus-4-5-20251101", tool="daydream", client=client)

    assert client.peak_active == _JUDGE_CONCURRENCY, f"peak={client.peak_active}"
    groups = json.loads(
        (model_results_dir(tmp_path, "claude-opus-4-5-20251101") / "dedup_groups.json").read_text()
    )
    assert list(groups.keys()) == urls
    expected = [[[0, 1]] if i != 4 else [[0], [1]] for i in range(12)]
    assert [groups[url]["daydream"] for url in urls] == expected


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


@pytest.mark.asyncio
async def test_direct_judge_records_per_pair_errors_in_source_order(tmp_path):
    """Judge failures are recorded per golden-candidate pair and re-emitted in
    source index order regardless of completion order; each errors entry carries
    the verbatim golden/candidate/error for its own pair."""
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "benchmark_data.json").write_text(
        json.dumps(
            {
                URL: {
                    "golden_comments": [
                        {"comment": "golden a", "severity": "medium"},
                        {"comment": "golden b", "severity": "medium"},
                    ],
                    "reviews": [
                        {
                            "tool": "daydream",
                            "repo_name": "repo",
                            "pr_url": URL,
                            "review_comments": [{"body": "the bug"}],
                        }
                    ],
                }
            }
        )
    )
    seed_candidates(tmp_path, model="claude-opus-4-5-20251101", tool="daydream",
                    texts=["cand a", "cand b", "cand c"])
    seed_dedup_groups(tmp_path, model="claude-opus-4-5-20251101", tool="daydream",
                      groups=[[0], [1], [2]])
    # Judge tasks run in (gi, ci) order: (0,0) ok, (0,1) fails, (0,2) ok,
    # (1,0) fails, (1,1) ok, (1,2) ok. OutOfOrderAnthropicJson completes them
    # in reverse, so the recorded errors must be re-sorted by source index when
    # persisted (2 errors / 6 comparisons stays under JUDGE_ERROR_RATIO_THRESHOLD).
    client = OutOfOrderAnthropicJson(
        [
            {"issues": ["cand a", "cand b", "cand c"]},
            {"groups": [[0], [1], [2]]},
            {"reasoning": "a0", "match": True, "confidence": 0.9},
            RuntimeError("judge failed for golden a x cand b"),
            {"reasoning": "a2", "match": True, "confidence": 0.7},
            RuntimeError("judge failed for golden b x cand a"),
            {"reasoning": "b1", "match": True, "confidence": 0.8},
            {"reasoning": "b2", "match": True, "confidence": 0.6},
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
    # Each failure is keyed to its own pair and re-emitted in source order.
    assert leaf["errors"] == [
        {"golden": "golden a", "candidate": "cand b", "error": "judge failed for golden a x cand b"},
        {"golden": "golden b", "candidate": "cand a", "error": "judge failed for golden b x cand a"},
    ]
    assert leaf["errors_count"] == 2
    # Failed pairs only record errors; successful pairs still count.
    assert (leaf["tp"], leaf["fp"], leaf["fn"]) == (2, 1, 0)
    assert [tp["golden_comment"] for tp in leaf["true_positives"]] == ["golden a", "golden b"]
    assert leaf["false_positives"] == [{"candidate": "cand c"}]
    assert scores.total_tp == 2


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            {
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
async def test_direct_judge_finds_two_tps_when_greedy_would_strand_one(tmp_path):
    """A greedy one-to-one pass can strand a golden whose candidate group was taken by an
    earlier high-confidence match even when a feasible two-TP assignment exists; the
    maximum-cardinality matching must find both true positives."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "benchmark_data.json").write_text(
        json.dumps(
            {
                URL: {
                    "golden_comments": [
                        {"comment": "bug a", "severity": "medium"},
                        {"comment": "bug b", "severity": "medium"},
                    ],
                    "reviews": [
                        {
                            "tool": "daydream",
                            "repo_name": "repo",
                            "pr_url": URL,
                            "review_comments": [{"body": "the bug"}],
                        }
                    ],
                }
            }
        )
    )
    seed_candidates(tmp_path, model="claude-opus-4-5-20251101", tool="daydream", texts=["fix c", "fix d"])
    seed_dedup_groups(tmp_path, model="claude-opus-4-5-20251101", tool="daydream", groups=[[0], [1]])
    # Greedy would take g1c0 (0.9) first, leaving g0 unmatched even though g0c0 + g1c1
    # is a feasible two-TP assignment. Judge tasks run in (gi, ci) order: g0c0, g0c1, g1c0, g1c1.
    client = FakeAnthropicJson(
        [
            {"issues": ["fix c", "fix d"]},
            {"groups": [[0], [1]]},
            {"reasoning": "g0c0", "match": True, "confidence": 0.5},
            {"reasoning": "g0c1", "match": False, "confidence": 0.0},
            {"reasoning": "g1c0", "match": True, "confidence": 0.9},
            {"reasoning": "g1c1", "match": True, "confidence": 0.5},
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
    assert [tp["golden_comment"] for tp in leaf["true_positives"]] == ["bug a", "bug b"]
    assert leaf["false_positives"] == []
    assert leaf["false_negatives"] == []
    assert (leaf["tp"], leaf["fp"], leaf["fn"]) == (2, 0, 0)
    assert leaf["precision"] == 1.0 and leaf["recall"] == 1.0
    assert (scores.total_tp, scores.total_fp, scores.total_fn) == (2, 0, 0)


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


class FailFastAnthropicJson:
    """Like FakeAnthropicJson but with real I/O-style awaits so pending semaphore
    jobs stay cancellable; tracks how many calls actually started."""

    def __init__(self, responses, fail_index=None):
        self._responses = list(responses)
        self._fail_index = fail_index
        self._next = 0
        self.started = 0

    async def complete_json(self, *, system, user, max_tokens):
        idx = self._next
        self._next += 1
        self.started += 1
        if self._fail_index is not None and idx == self._fail_index:
            await asyncio.sleep(0.001)
            raise RuntimeError("permanent 5xx")
        await asyncio.sleep(0.02)
        return self._responses[idx]


@pytest.mark.asyncio
async def test_direct_extraction_fails_fast_and_cancels_pending(tmp_path):
    """Extraction is fail-fast: a permanent failure aborts and cancels pending
    jobs so no further paid calls start after the abort (nothing is written)."""
    urls = [f"https://x/pull/{i}" for i in range(1, 13)]  # 12 URLs
    reviews_by_url = {url: [f"review for {url}"] for url in urls}
    seed_parallel_benchmark_data(tmp_path, tool="daydream", reviews_by_url=reviews_by_url)

    responses = [{"issues": [f"ok {i}"]} for i in range(12)]
    client = FailFastAnthropicJson(responses, fail_index=5)  # 6th job fails

    with pytest.raises(RuntimeError):
        await run_anthropic_extraction(
            tmp_path, "claude-opus-4-5-20251101", tool="daydream", client=client
        )

    # A permanent failure must not schedule every remaining paid call.
    assert client.started < len(urls), f"fail-fast broken: started {client.started}/{len(urls)}"
    # Nothing is persisted after an abort.
    cand = model_results_dir(tmp_path, "claude-opus-4-5-20251101") / "candidates.json"
    assert not cand.exists(), "fail-fast abort must not write candidates.json"
