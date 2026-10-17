#!/usr/bin/env python3
"""Validate analyst evidence and summarize frozen cohorts; never execute captured data.

Counts establish observed recurrence, not the correctness of causal judgments.
All outputs go to stdout; inputs and source captures remain unchanged.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from cohort import canonical, digest, iter_runs, load_manifest, parse_json

CATEGORIES = {"environment_scope", "investigation_loop", "handoff_loss", "verification_gap",
              "budget_allocation", "finding_quality", "backend_lifecycle", "capture_failure"}
MAX_TEXT = 2000
MAX_EVIDENCE = 6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    value = parse_json(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: expected a JSON object")
    return value


def text(value, name, limit=MAX_TEXT, empty=False):
    require(isinstance(value, str) and (empty or bool(value.strip())) and len(value) <= limit,
            f"{name}: expected {'possibly empty ' if empty else 'nonempty '}text <= {limit} characters")
    return value


def positive(value, name):
    require(type(value) is int and value > 0, f"{name}: expected a positive integer")
    return value


def pattern_name(value):
    name = text(value, "pattern", 80)
    require(bool(re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name)), "pattern must be a lowercase stable slug")
    return name


def strings(value, name):
    require(isinstance(value, list) and value, f"{name}: expected a nonempty list")
    for item in value:
        text(item, name)
    require(len(set(value)) == len(value), f"{name}: duplicate values")
    return value


def load_inputs(manifest_path, plan_path):
    manifest = load_manifest(manifest_path)
    plan = read_json(plan_path)
    require(plan.get("schema") == "daydream.trace-batches.v1", "unsupported batch plan schema")
    require(plan.get("cohort_id") == manifest["cohort_id"], "batch plan cohort_id does not match manifest")
    require(plan.get("plan_id") == digest({key: value for key, value in plan.items() if key != "plan_id"}),
            "batch plan plan_id does not match its contents; regenerate the plan explicitly")
    text(plan.get("question"), "question")
    positive(plan.get("evidence_chars"), "evidence_chars")
    positive(plan.get("max_observations"), "max_observations")
    require(isinstance(plan.get("batches"), list) and plan["batches"], "plan requires batches")
    runs = {row["run_id"]: row for row in manifest["runs"]}
    batches, assigned = {}, set()
    for batch in plan["batches"]:
        require(isinstance(batch, dict), "batch must be an object")
        batch_id = text(batch.get("batch_id"), "batch_id", 100)
        require(batch_id not in batches, f"duplicate batch {batch_id}")
        ids = strings(batch.get("run_ids"), f"{batch_id}.run_ids")
        require(not (set(ids) - runs.keys()), f"{batch_id}: assignment contains an unknown run")
        require(not (set(ids) & assigned), f"{batch_id}: duplicate run assignment")
        require(type(batch.get("bytes")) is int and batch["bytes"] == sum(runs[r]["bytes"] for r in ids),
                f"{batch_id}: bytes do not match frozen run assignments")
        assigned.update(ids)
        batches[batch_id] = batch
    require(assigned == runs.keys(), "plan assignment must cover every cohort run exactly once")
    return manifest, plan, runs, batches


def pointer_value(record, pointer):
    text(pointer, "pointer", empty=True)
    require(not pointer or pointer.startswith("/"), "pointer must be an RFC 6901 path")
    value = record
    try:
        for token in pointer[1:].split("/") if pointer else []:
            require(not re.search(r"~(?![01])", token), f"invalid pointer escape: {pointer}")
            key = token.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                require(bool(re.fullmatch(r"0|[1-9][0-9]*", key)), f"invalid pointer array index: {pointer}")
                value = value[int(key)]
            else:
                value = value[key]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"pointer does not resolve: {pointer}") from exc
    return value


def evidence(rows, record, chars):
    require(isinstance(rows, list), "evidence and counterevidence must be lists")
    seen = set()
    for row in rows:
        require(isinstance(row, dict), "evidence must be an object")
        pointer = row.get("pointer")
        value = pointer_value(record, pointer)
        excerpt = text(row.get("excerpt"), "evidence excerpt", chars)
        rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
        require(excerpt in rendered, f"evidence excerpt does not occur at pointer {pointer}")
        identity = (pointer, excerpt)
        require(identity not in seen, "duplicate evidence pointer/excerpt")
        seen.add(identity)


def validate_report(report, manifest, plan, batches):
    require(report.get("schema") == "daydream.trace-report.v1", "unsupported report schema")
    require(not report.get("template"), "report is an unfinished template; fill it and remove template marker")
    require(report.get("cohort_id") == manifest["cohort_id"], "report cohort_id does not match manifest")
    require(report.get("plan_id") == plan["plan_id"], "report plan_id does not match the investigation plan")
    batch_id = report.get("batch_id")
    require(isinstance(batch_id, str) and batch_id in batches, "report batch_id is not assigned by plan")
    require(isinstance(report.get("runs"), list), "report runs must be a list")
    assigned = set(batches[batch_id]["run_ids"])
    seen = set()
    for row in report["runs"]:
        require(isinstance(row, dict), "reported run must be an object")
        require(set(row) == {"run_id", "status", "reason", "assessments"},
                "reported run accepts only run_id/status/reason/assessments; frozen metadata belongs to the manifest")
        run_id = row.get("run_id")
        require(isinstance(run_id, str) and run_id in assigned, "report run violates batch assignment")
        require(run_id not in seen, f"duplicate reported run {run_id}")
        seen.add(run_id)
        require(row.get("status") in {"analyzed", "skipped", "unavailable"}, "invalid run status")
        text(row.get("reason"), "reason", empty=row["status"] == "analyzed")
        assessments = row.get("assessments")
        require(isinstance(assessments, list), "assessments must be a list")
        require(len(assessments) <= plan["max_observations"], "run exceeds plan max_observations")
        require(row["status"] == "analyzed" or not assessments, "skipped/unavailable runs cannot have assessments")
        patterns = set()
        for item in assessments:
            require(isinstance(item, dict), "assessment must be an object")
            pattern = pattern_name(item.get("pattern"))
            require(pattern not in patterns, f"duplicate pattern {pattern} within run {run_id}")
            patterns.add(pattern)
            require(item.get("category") in CATEGORIES, f"unknown category for {pattern}")
            verdict, occurrences = item.get("verdict"), item.get("occurrences")
            require(verdict in {"present", "absent", "unknown"}, "invalid assessment verdict")
            if verdict == "unknown":
                require(occurrences is None, "unknown occurrences must be null, never zero")
            elif verdict == "present":
                positive(occurrences, "present occurrences")
            else:
                require(type(occurrences) is int and occurrences == 0, "absent occurrences must be zero")
            text(item.get("observation"), "observation")
            text(item.get("intervention"), "intervention", empty=verdict != "present")
            require(item.get("confidence") in {"high", "medium", "low"}, "invalid analyst confidence")
            support, counter = item.get("evidence"), item.get("counterevidence")
            require(isinstance(support, list) and isinstance(counter, list),
                    "evidence and counterevidence must be lists")
            require(len(support) + len(counter) <= MAX_EVIDENCE,
                    f"at most {MAX_EVIDENCE} evidence entries per assessment")
            require(verdict != "present" or support, "present assessment requires evidence")
            require(verdict != "absent" or support or counter, "absent assessment requires evidence/counterevidence")
    return batch_id


def template(manifest, plan, batches, batch_id):
    require(batch_id in batches, "--batch is not in plan")
    # All frozen sources are checked, including runs assigned to other batches.
    for _, record in iter_runs(manifest["runs"]):
        del record
    return {"schema": "daydream.trace-report.v1", "cohort_id": manifest["cohort_id"], "plan_id": plan["plan_id"],
            "batch_id": batch_id, "template": True,
            "runs": [{"run_id": run_id, "status": "skipped", "reason": "TODO: analyze this run",
                      "assessments": []} for run_id in batches[batch_id]["run_ids"]]}


def assessment_template(pattern, category):
    return {"pattern": pattern_name(pattern), "category": category, "verdict": "unknown", "occurrences": None,
            "observation": None, "confidence": "low", "evidence": [], "counterevidence": [], "intervention": ""}


def pattern_counts(run_ids, rows, pattern):
    groups = {key: [] for key in ("affected_runs", "assessable_runs", "counterexample_runs", "unknown_runs",
                                  "unassessed_runs", "skipped_runs", "unavailable_runs", "missing_runs")}
    occurrences = 0
    for run_id in run_ids:
        row = rows.get(run_id)
        if row is None:
            groups["missing_runs"].append(run_id)
            continue
        if row["status"] != "analyzed":
            groups[f"{row['status']}_runs"].append(run_id)
            continue
        item = next((item for item in row["assessments"] if item["pattern"] == pattern), None)
        if item is None:
            groups["unassessed_runs"].append(run_id)
        elif item["verdict"] == "unknown":
            groups["unknown_runs"].append(run_id)
        else:
            groups["assessable_runs"].append(run_id)
            if item["verdict"] == "present":
                groups["affected_runs"].append(run_id)
                occurrences += item["occurrences"]
            else:
                groups["counterexample_runs"].append(run_id)
    return {**groups, "affected_count": len(groups["affected_runs"]),
            "assessable_count": len(groups["assessable_runs"]), "observed_occurrences": occurrences}


def synthesize(manifest, plan, runs, batches, paths):
    rows, reported_batches, categories = {}, set(), {}
    for path in paths:
        report = read_json(path)
        batch_id = validate_report(report, manifest, plan, batches)
        require(batch_id not in reported_batches, f"duplicate report for batch {batch_id}")
        reported_batches.add(batch_id)
        for row in report["runs"]:
            rows[row["run_id"]] = row
            for item in row["assessments"]:
                pattern, category = item["pattern"], item["category"]
                require(pattern not in categories or categories[pattern] == category,
                        f"pattern {pattern} has conflicting categories; reconcile reports before synthesis")
                categories[pattern] = category
    # Validate every frozen source once; retain only analyst and manifest metadata.
    for frozen, record in iter_runs(runs.values()):
        for item in rows.get(frozen["run_id"], {}).get("assessments", []):
            evidence(item["evidence"], record, plan["evidence_chars"])
            evidence(item["counterevidence"], record, plan["evidence_chars"])
        del record
    strata = {}
    for run_id, row in runs.items():
        context = row["context"]
        stratum_id = hashlib.sha256(canonical(context)).hexdigest()
        strata.setdefault(stratum_id, {"stratum_id": stratum_id, "context": context, "run_ids": []})
        strata[stratum_id]["run_ids"].append(run_id)
    ordered_strata = sorted(strata.values(), key=lambda row: row["stratum_id"])
    patterns = []
    for pattern, category in sorted(categories.items()):
        patterns.append({"pattern": pattern, "category": category,
                         "population": "aggregate across explicit context strata",
                         **pattern_counts(runs, rows, pattern),
                         "strata": [{"stratum_id": s["stratum_id"], **pattern_counts(s["run_ids"], rows, pattern)}
                                    for s in ordered_strata]})
    missing_batches = sorted(batches.keys() - reported_batches)
    missing_runs = [run_id for run_id in runs if run_id not in rows]
    return {"schema": "daydream.trace-synthesis.v1", "cohort_id": manifest["cohort_id"], "plan_id": plan["plan_id"],
            "question": plan["question"], "selection": manifest["selection"], "exclusions": manifest["exclusions"],
            "limits": {key: plan[key] for key in ("max_observations", "evidence_chars")},
            "complete": not missing_batches and not missing_runs and bool(patterns)
            and all(row["status"] == "analyzed" for row in rows.values()),
            "missing_batches": missing_batches, "missing_runs": missing_runs,
            "strata": ordered_strata, "patterns": patterns,
            "runs": [{**row, **rows.get(run_id, {"status": "missing", "reason": "No assigned report entry",
                                               "assessments": []})} for run_id, row in runs.items()],
            "interpretation": "Evidence membership and counts validated; "
            "confidence, cause, impact and priority require adjudication."}


def experiment_template(synthesis):
    require(synthesis.get("schema") == "daydream.trace-synthesis.v1", "expected a trace synthesis")
    text(synthesis.get("cohort_id"), "cohort_id")
    conditions = {"backend": None, "model": None, "profile": None, "reasoning_effort": None, "budgets": {},
                  "workloads": [], "producer_revision": None}
    return {"schema": "daydream.trace-experiment.v1", "cohort_id": synthesis["cohort_id"], "template": True,
            "change": {"pattern": None, "description": None}, "predicted_effect": None,
            "baseline": dict(conditions), "candidate": dict(conditions), "repeats": None,
            "discovery_workloads": [], "heldout_workloads": [], "quality_guardrails": [], "rejection_criterion": None}


def validate_experiment(packet):
    require(packet.get("schema") == "daydream.trace-experiment.v1", "unsupported experiment schema")
    require(not packet.get("template"), "experiment is an unfinished template; fill it and remove template marker")
    text(packet.get("cohort_id"), "cohort_id")
    change = packet.get("change")
    require(isinstance(change, dict) and set(change) == {"pattern", "description"},
            "change must describe exactly one pattern and one harness change")
    pattern_name(change.get("pattern"))
    text(change.get("description"), "change.description")
    text(packet.get("predicted_effect"), "predicted_effect")
    discovery = set(strings(packet.get("discovery_workloads"), "discovery_workloads"))
    heldout = set(strings(packet.get("heldout_workloads"), "heldout_workloads"))
    require(not discovery & heldout, "discovery_workloads and heldout_workloads must be disjoint")
    conditions = []
    for name in ("baseline", "candidate"):
        condition = packet.get(name)
        require(isinstance(condition, dict), f"{name}: expected conditions object")
        for key in ("backend", "model", "profile", "reasoning_effort", "producer_revision"):
            text(condition.get(key), f"{name}.{key}")
        budgets = condition.get("budgets")
        require(isinstance(budgets, dict) and budgets, f"{name}.budgets must record explicit limits")
        for key, value in budgets.items():
            text(key, f"{name}.budgets key", 100)
            require(value is None or (type(value) in {int, float} and value >= 0),
                    f"{name}.budgets: use nonnegative numbers or null for an explicitly unbounded limit")
        require(set(strings(condition.get("workloads"), f"{name}.workloads")) == discovery | heldout,
                f"{name}.workloads must equal discovery and heldout workload union")
        conditions.append(condition)
    require(conditions[0].keys() == conditions[1].keys(), "matched baseline/candidate condition fields required")
    # Workload sets were matched above; all other declared conditions must match exactly.
    for key in sorted(conditions[0].keys() - {"producer_revision", "workloads"}):
        require(canonical(conditions[0][key]) == canonical(conditions[1][key]),
                f"matched baseline/candidate {key} required")
    require(type(packet.get("repeats")) is int and packet["repeats"] >= 2, "repeats must be an integer >= 2")
    strings(packet.get("quality_guardrails"), "quality_guardrails")
    text(packet.get("rejection_criterion"), "rejection_criterion")
    return {"schema": "daydream.trace-experiment-validation.v1", "valid": True,
            "cohort_id": packet["cohort_id"], "execution": "not executed; proposal requires human review"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("template", "synthesize"):
        command = commands.add_parser(name)
        command.add_argument("manifest")
        command.add_argument("plan")
        if name == "template":
            command.add_argument("--batch", required=True)
        else:
            command.add_argument("reports", nargs="*")
    commands.add_parser("experiment-template").add_argument("synthesis")
    commands.add_parser("validate-experiment").add_argument("packet")
    assessment = commands.add_parser("assessment-template")
    assessment.add_argument("--pattern", required=True)
    assessment.add_argument("--category", required=True, choices=sorted(CATEGORIES))
    args = parser.parse_args()
    try:
        if args.command in {"template", "synthesize"}:
            manifest, plan, runs, batches = load_inputs(args.manifest, args.plan)
            output = template(manifest, plan, batches, args.batch) if args.command == "template" else synthesize(
                manifest, plan, runs, batches, args.reports)
        elif args.command == "experiment-template":
            output = experiment_template(read_json(args.synthesis))
        elif args.command == "assessment-template":
            output = assessment_template(args.pattern, args.category)
        else:
            output = validate_experiment(read_json(args.packet))
        print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
