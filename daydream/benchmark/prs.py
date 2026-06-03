"""Pinned registry of the 26 evaluable benchmark PRs.

The withmartian ``code-review-benchmark`` offline set keys each PR by its
upstream GitHub URL (the ``benchmark_data.json`` dict key). This module pins
the 26 covered-3 (Python/Go/TypeScript) PRs daydream is scored against, with
their base/head commit SHAs transcribed as literals from the verified
resolution research — never re-resolved over the network.

Scope (per spec "Sentry evaluable set" and ``research/grafana-calcom-shas.md``):
- Sentry (Python): 6 reconstructable ``getsentry/sentry`` PRs (4 greptile
  mirrors / synthetic excluded).
- Grafana (Go): 10 / 10 evaluable.
- Cal.com (TypeScript): 10 / 10 evaluable.

Exports:
- ``EvaluablePR`` — frozen dataclass describing one pinned PR.
- ``EVALUABLE_PRS`` — module-level tuple of all 26.
- ``load_evaluable_prs()`` — accessor returning ``EVALUABLE_PRS``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluablePR:
    """One pinned evaluable benchmark PR.

    Attributes:
        golden_url: Upstream GitHub PR URL; the ``benchmark_data.json`` dict key.
        clone_url: HTTPS clone URL of the upstream repository.
        source_repo: Logical repo name as stored in ``benchmark_data.json``
            (``"sentry"``, ``"grafana"``, ``"cal.com"``).
        pr_number: Upstream pull-request number.
        base_sha: Full 40-char hex base commit SHA (daydream's diff base).
        head_sha: Full 40-char hex head commit SHA (``pull/<N>/head``).
    """

    golden_url: str
    clone_url: str
    source_repo: str
    pr_number: int
    base_sha: str
    head_sha: str


_SENTRY_CLONE_URL = "https://github.com/getsentry/sentry"
_GRAFANA_CLONE_URL = "https://github.com/grafana/grafana"
_CALCOM_CLONE_URL = "https://github.com/calcom/cal.com"


def _sentry(pr_number: int, base_sha: str, head_sha: str) -> EvaluablePR:
    return EvaluablePR(
        golden_url=f"{_SENTRY_CLONE_URL}/pull/{pr_number}",
        clone_url=_SENTRY_CLONE_URL,
        source_repo="sentry",
        pr_number=pr_number,
        base_sha=base_sha,
        head_sha=head_sha,
    )


def _grafana(pr_number: int, base_sha: str, head_sha: str) -> EvaluablePR:
    return EvaluablePR(
        golden_url=f"{_GRAFANA_CLONE_URL}/pull/{pr_number}",
        clone_url=_GRAFANA_CLONE_URL,
        source_repo="grafana",
        pr_number=pr_number,
        base_sha=base_sha,
        head_sha=head_sha,
    )


def _calcom(pr_number: int, base_sha: str, head_sha: str) -> EvaluablePR:
    return EvaluablePR(
        golden_url=f"{_CALCOM_CLONE_URL}/pull/{pr_number}",
        clone_url=_CALCOM_CLONE_URL,
        source_repo="cal.com",
        pr_number=pr_number,
        base_sha=base_sha,
        head_sha=head_sha,
    )


# Sentry — 6 reconstructable getsentry/sentry PRs (spec "Sentry evaluable set";
# research/sentry-pr-resolution.md INCLUDE table). The 4 greptile-mirror /
# synthetic entries are deliberately excluded.
_SENTRY_PRS: tuple[EvaluablePR, ...] = (
    _sentry(67876, "344aa102e7818606e29426ebe69d5a680d8727c6", "bb75657fc8f13923c1d7983f422290908a1e7310"),
    _sentry(93824, "de11fb0166a7244115bb066edd65ec0d6b7e365c", "3162ad68a5c87666788b27a44eb31235025091a9"),
    _sentry(77754, "bb5a6837cb5b3d8d3b174e17d42ec14486ef8738", "9501091c52ae94e8d916f79b35d21975b3f9cadb"),
    _sentry(80528, "0cfc28e76ddc986d2d89dd9b9f63ee916a18a5f9", "dcdcadb771128e79259cc9eff9c70c38fc597976"),
    _sentry(95633, "fd358e8a388939959369f08f616e29552cdaf96e", "9966ec5a13e331659c3ea00981f9b11b0faf821f"),
    _sentry(80168, "bdd229e3f22e307fe40b30ef99e92ff3f6723da4", "8422030ef456e3a898415e96475b4d8ddfc7640f"),
)

# Grafana — 10 / 10 evaluable (research/grafana-calcom-shas.md Grafana table).
_GRAFANA_PRS: tuple[EvaluablePR, ...] = (
    _grafana(79265, "50f4e78a39914711d0d231a501ba215becf17ebc", "bbd8c507cdf56f6b884373809e0b24abd4a2353d"),
    _grafana(103633, "5634ca44f799a82161f0401dadc286452f3bdbf8", "7562f37880367411a62304bcdcdc178bced23906"),
    _grafana(76186, "58ba11ecbd60863e2bfc6e32f0399ed4feb3927a", "303cdc2caf256aa5df660611120c2e3f7392365d"),
    _grafana(107534, "d9a8253640464014aa3662c91131278990cfb828", "1cab4143d443c33079566f6ca36931eb8468159f"),
    _grafana(106778, "47e5bd23163e6037fc0741068ac78930bba2769f", "8df850371034b73dc4dd9908cc30fa09f12a1f97"),
    _grafana(90045, "66c4dff17e9147b11fb9b14c6ed63089a599da65", "e369f24665ec70e1e3700f457d25ef83301d931a"),
    _grafana(80329, "a886bd3c79a417a70b51509384d1f1ec3e87e96b", "04cfa3bfd469c035ff8b35f9a66867c4e8d5dcf4"),
    _grafana(90939, "3ce1a5b0caab215346a51629b0344b90d67e9478", "b1613e320acff00623e6efc59f00ee68c7684a97"),
    _grafana(94942, "cbe1e7d63f098e306058c0fbcab2f5c30602fa7d", "f3317b329b4eb8fd96f99dd86525bc4a22d20248"),
    _grafana(97529, "871af07203177ce59a24e742107119409274f48a", "26fed312840cd76b766cbd2158e17a7e6c0ec548"),
)

# Cal.com — 10 / 10 evaluable (research/grafana-calcom-shas.md Cal.com table).
_CALCOM_PRS: tuple[EvaluablePR, ...] = (
    _calcom(8087, "ba9688a04a8398c9a8332ee7061bfae2f2efd524", "820d7fa87e0c824c5dee8082a33efadb1a2566f7"),
    _calcom(10600, "efa6d464a38e60ceeeb88f40668c1c4ac4bfaf54", "54486a059cd2032042189bb565646ba4e0f6bd61"),
    _calcom(10967, "a308075bc39b77ed7059b0cae9d443d669a7bf98", "de628295646d0848226618108a52f2f1e5d04ac0"),
    _calcom(22345, "c9a47dd0cebd121ff73f24e1f8c1829579daf8e3", "da09bc0808e7c6df1bc10e00d5acf764d94c7cc7"),
    _calcom(7232, "d1440bb5d2f2190ad7ae1d26c165f970055b5370", "6048e2a86b50e81e1e3b1b467dfea5a895add3dc"),
    _calcom(8330, "93cb21f55a2a40d9116c014741f5679fbeff5fd9", "ee38fd295fd294b9fc787eba482bde24bbfea69b"),
    _calcom(11059, "bc89fe00ea84d20bedcec782f0701b9711dc8201", "9fde0e906897cc0f4f71793f647dd629faba3317"),
    _calcom(14943, "917c7b0764f4c6bfdca95db2c5f73dbfdbe11757", "c790227e0cda780e6fea9bb03af27948d9e286b9"),
    _calcom(14740, "b004587262e8221083bafbe9a0c515e7becaa7b3", "92f44dcea7ff19e9123a30c63c167a2938df5a55"),
    _calcom(22532, "30a92a4d66b417f7c19dee2b343099adc6e88aa6", "5fd11f9faa79c4aefd39975d1b0e963e5034f793"),
)

EVALUABLE_PRS: tuple[EvaluablePR, ...] = _SENTRY_PRS + _GRAFANA_PRS + _CALCOM_PRS


def load_evaluable_prs() -> tuple[EvaluablePR, ...]:
    """Return the pinned tuple of all 26 evaluable benchmark PRs.

    Returns:
        The module-level ``EVALUABLE_PRS`` tuple (6 Sentry, 10 Grafana,
        10 Cal.com), with SHAs pinned as literals from the resolution research.
    """
    return EVALUABLE_PRS
