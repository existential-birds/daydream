# Independent final mutation-evidence review

Reviewer backend_audit; read-only, no source edits or test executions. Reviewed all 31 recorded controls and all 62 final mutated/restored logs; compared published mutation replacement metadata to local result JSON. Inspected isolated checkout /private/tmp/daydream-owner-controls using git diff -- daydream (empty) and computed SHA256 for every one of the 15 actual production owner files; all match recorded restored hashes. Candidate full make check was running separately; no candidate edits occurred in this review.

**Final verdict: APPROVE all 31 controls. No unresolved evidence gap.** All31 final control records have mutated_exit1/restored_exit0; all35 selected keeper executions fail at their intended behavioral oracles and all35 restored executions pass. The refined extraction control suppresses only agent_message text, preserving the earlier reasoning assertions. Added MetricsEvent cardinality control independently removes the first34594-token event while preserving CostEvent emission; the golden keeper fails count1vs2. Exact final metadata agrees between local and published JSON.

Confirmed intended behavior failures:

| Control | Actual failed oracle |
| --- | --- |
| projection-route | CLI exit1 due wrong out-directory projection missing _SUCCESS; real input-routing refusal, not collection error. |
| per-finding-label | Transferred outcome_label != contested assertion. |
| adjudication-order | Full reversed-observation comparison differs in disposition and labeler. |
| cache-fresh-hit | Explicit fresh second read returns n2 instead of n1. |
| gate-ratio | Transferred accepted_ratio <=1 fails on2.0. |
| queue-rejected | Complete queue omits rejected outcome. |
| queue-ambiguous | Default queue duplicates ambiguous outcome. |
| recorder-label | Persisted extra.daydream_run_flow review instead of normal. |
| codex-legacy-patch | Exact patch.input action corrupted instead of modified. |
| codex-provenance | Native MetricsEvent measurement_source terminal instead of turn_end. |
| codex-cost-provenance | CostEvent source reported instead of estimated. |
| codex-extraction | Refined agent_message-only mutation: earlier thinking assertions pass; top-level structured outcome None fails at exact issues object; simple text len0; output-text-blocks structured outcome None. All3 intended catches. |
| supervisor-command | Actual extension sees cd-prefix command instead of make test. |
| xcrun-parent | Actual resolver returns executable instead of parent path. |
| codex-positive-turn-tokens | Native positive prompt-token assertion fails at0; name now describes positivity precisely. |
| codex-metrics-cardinality | First34594-token native MetricsEvent omitted; trailing CostEvent remains; exact len2 assertion fails at1. |
| codex-final-total | Persisted final usage141188 != sum70594. |
| pi-native-message | Required native TextEvent absent; readonly parity's actual Step.message empty. Both selected nodes fail behaviorally. |
| osprey-turn-native | Actual TurnEnd model_source configured instead of native. |
| osprey-session-usage | Actual CostEvent measurement_source turn_end instead of session. |
| claude-background-switch | Delivered SDK child environment switch0 instead of1. |
| claude-bash-default | Delivered default timeout600000 instead of3600000. |
| claude-bash-max | Delivered max timeout600000 instead of3600000. |
| claude-malformed-bash | Registered callback chain gives no deny for missing-command Bash; KeyError at deny decision assertion reflects absent behavioral output. |
| claude-deny-wording | Registered actual Write denial says read-only summarizer, losing shared guard wording. |
| diagram-wrapper | Both literal fixture wrappers differ h2 vs h3 after Mermaid golden assertions pass. |
| diagram-order | Sequence-first ordering assertion fails. |
| diagram-immutable-spec | Deep-copied input snapshot equality fails after actual renderer adds mutation key. |
| diagram-report-insertion | Actual report lacks transferred Diagrams section. |
| python-branch-classifier | Actual grammar classifier omits literal elif line8. |
| python-terminal-classifier | Actual grammar classifier omits literal return line42. |

No final log shows collection/import/syntax failure as the claimed catch. All restored logs have ordinary pass summaries, no skips or xfails. Replacements target actual production owners; fixture/test code unchanged during controls. Initial same-length stale bytecode attempt and initial malformed-Bash command-None mutation were unsuccessful drafts, not final evidence; clearing isolated owner __pycache__ around controls and correcting missing-command discrimination to command==emptystring led to the intended final behavioral failure. Exact restored owner hashes and empty git diff show no production mutation persists.

No performance result is inferred from these narrow controls, and no global gate result is asserted by this reviewer.

Final refresh: all62 final log artifacts read/checked; all31 controls match expected selected failure and restored pass cardinalities. Recomputed all15 actual owner file SHA256 values after refined controls: all match recorded exact restored bytes. Rechecked git diff -- daydream in /private/tmp/daydream-owner-controls: empty. Initial broad extraction control was superseded by the agent_message-only final control; it is not counted as proof. No source edit or test execution by this reviewer.
