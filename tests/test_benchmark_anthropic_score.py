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


class FakeAnthropicJson:
    def __init__(self, responses):
        self._responses = list(responses)

    async def complete_json(self, *, system, user, max_tokens):
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


@pytest.mark.asyncio
async def test_direct_dedup_writes_groups_and_falls_back_to_singletons(tmp_path):
    seed_candidates(tmp_path, model="claude-opus-4-5-20251101", tool="daydream", texts=["same bug", "same issue"])
    client = FakeAnthropicJson([{"groups": [[0, 1]]}])

    await run_anthropic_dedup(tmp_path, "claude-opus-4-5-20251101", tool="daydream", client=client)

    groups = json.loads((model_results_dir(tmp_path, "claude-opus-4-5-20251101") / "dedup_groups.json").read_text())
    assert groups[URL]["daydream"] == [[0, 1]]


@pytest.mark.asyncio
async def test_direct_dedup_invalid_response_uses_singletons(tmp_path):
    seed_candidates(tmp_path, model="claude-opus-4-5-20251101", tool="daydream", texts=["a", "b"])
    client = FakeAnthropicJson([{"groups": [[0, 0]]}])

    await run_anthropic_dedup(tmp_path, "claude-opus-4-5-20251101", tool="daydream", client=client)

    groups = json.loads((model_results_dir(tmp_path, "claude-opus-4-5-20251101") / "dedup_groups.json").read_text())
    assert groups[URL]["daydream"] == [[0], [1]]


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
