"""Operator workflow: frozen plans, native generation, supplied judging/reports."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import anyio

from daydream import git_ops
from daydream.benchmark.reviewbench.contract import (
    Case,
    execution_accounting,
    file_digest,
    load_cases,
    verify_source,
)
from daydream.commands import review
from daydream.git_ops.process import _run_git
from daydream.review_profile import ProfileError
from daydream.run_artifacts import _resolve_review_profile
from daydream.run_config import RunConfig


def candidate_identity() -> dict[str, Any]:
    """Identify the executing package bytes, including editable candidate changes."""
    package = Path(__file__).resolve().parents[2]
    files = sorted(path for path in package.rglob('*') if path.is_file()
                   and '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.pyo'}
                   and path.name != '.DS_Store')
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(package)).encode() + b'\0' + path.read_bytes() + b'\0')
    try:
        revision: str | None = git_ops.head_sha(package)
        dirty: bool | None = bool(_run_git(package, ['status', '--porcelain', '--untracked-files=all', '--', '.'],
                                           error_context='candidate source status failed').stdout)
    except git_ops.GitError:
        revision = None
        dirty = None
    return {'version': importlib.metadata.version('daydream'), 'revision': revision,
            'source_dirty': dirty, 'artifact_sha256': digest.hexdigest()}


def _write_new(path: Path, value: Any) -> None:
    """Install absent output only; competing/prior bytes must survive."""
    payload = (json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}-') as staging:
        staging.write(payload)
        staging.flush()
        os.fsync(staging.fileno())
        os.link(staging.name, path, follow_symlinks=False)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add one namespace without changing the private Harbor commands."""
    parser = subparsers.add_parser('reviewbench', help='supplied ReviewBench through the native reviewer')
    commands = parser.add_subparsers(dest='reviewbench_command', required=True)
    plan = commands.add_parser('plan', help='record supplied membership and candidate; no models')
    plan.add_argument('directory', type=Path)
    plan.add_argument('--source', type=Path, required=True, help='checkout at the documented pinned revision')
    plan.add_argument('--candidate', required=True, help='candidate label, used as an isolated directory')
    plan.add_argument('--producer', required=True)
    plan.add_argument('--set', choices=('test', 'full'), default='test')
    plan.add_argument('--rounds', type=int, choices=(1, 3), default=1)
    for verb in ('run', 'status', 'judge', 'report'):
        command = commands.add_parser(verb)
        command.add_argument('directory', type=Path, help='candidate directory created by plan')
        if verb != 'report':
            command.add_argument('--round', type=int, default=1)
        if verb == 'run':
            command.add_argument('--pr', type=int, help='zero-based supplied set index; default all cases')
            command.add_argument('--repository', type=Path, help='local frozen repository for a single --pr')
            command.epilog = 'Append -- followed by ordinary native review flags and configuration root.'
        if verb == 'judge':
            command.add_argument('--provider', required=True)
            command.add_argument('--model', required=True)
            command.add_argument('--concurrency', type=int, default=1)


def _plan(args: argparse.Namespace) -> int:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.candidate) or not args.producer.strip():
        raise ValueError('candidate must be a simple directory label and producer must be nonempty')
    source = args.source.resolve()
    provenance = verify_source(source)
    cases = load_cases(source, args.set)
    directory = args.directory.resolve() / args.candidate
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 1, 'benchmark': provenance, 'source': str(source),
                'set': args.set, 'rounds': args.rounds, 'producer': args.producer,
                'candidate': candidate_identity(), 'candidate_label': args.candidate,
                'cases': [asdict(case) for case in cases]}
    _write_new(directory / 'plan.json', manifest)
    print(directory)
    return 0


def _load(directory: Path) -> tuple[dict[str, Any], list[Case]]:
    plan = json.loads((directory / 'plan.json').read_text())
    if not isinstance(plan, dict) or plan.get('schema_version') != 1 or plan.get('rounds') not in (1, 3):
        raise ValueError('unsupported ReviewBench plan')
    source = Path(plan['source'])
    if verify_source(source) != plan['benchmark']:
        raise ValueError('ReviewBench source differs from frozen plan')
    cases = [Case.from_dict(raw) for raw in plan['cases']]
    if cases != load_cases(source, plan['set']):
        raise ValueError('planned membership differs from supplied set')
    return plan, cases


def _round(directory: Path, number: int, plan: dict[str, Any]) -> Path:
    if number < 1 or number > plan['rounds']:
        raise ValueError('round is outside this plan')
    return directory / f'round-{number}'


def _accounting(directory: Path, cases: list[Case], source: Path) -> dict[str, Any]:
    outcomes = {path.stem: json.loads(path.read_text()) for path in (directory / 'outcomes').glob('*.json')}
    if any(not isinstance(outcome, dict) for outcome in outcomes.values()):
        raise ValueError('ReviewBench outcomes must be objects')
    produced = sorted(path.stem for path in (directory / 'findings').glob('*.json'))
    expected = {case.key: case for case in cases}
    for key in produced:
        path = directory / 'findings' / f'{key}.json'
        outcome = outcomes.get(key, {})
        if (key not in expected or outcome.get('status') not in {'complete', 'completed_empty'}
                or outcome.get('candidate_sha256') != file_digest(path)
                or Case.from_dict(json.loads(path.read_text())['pr']) != expected[key]):
            raise ValueError(f'unpaired, changed or mismatched candidate artifact: {key}')
    evaluation = directory / 'evaluation'
    evaluated = None
    if evaluation.exists():
        from daydream.benchmark.reviewbench.evaluator import load_evaluation

        verified = load_evaluation(source, directory / 'findings', evaluation)
        evaluated = [row['pr_key'] for row in verified['metrics']['per_pr']]
    return execution_accounting([case.key for case in cases], outcomes, produced, evaluated)


def _native_config(args: list[str]) -> RunConfig:
    if args[:1] == ['--']:
        args = args[1:]
    config = review._parse_args(['--review', '--non-interactive', '--no-archive', '--no-eval',
                                 '--no-capture-data', *args])
    if (config.base or config.branch or config.findings_out or config.flow_name or config.start_at != 'review'
            or config.dump_artifacts or config.trajectory_path or config.trajectory_hub_repo
            or config.dataset_store_path or config.dataset_capture):
        raise ValueError('benchmark owns source/output and disables publication, resumes and dataset capture')
    root = Path(config.target or Path.cwd()).resolve()
    profile = config.review_profile_path
    if profile is None and config.file_config is not None:
        profile = config.file_config.review_profile
        if profile is not None:
            profile = root / profile
    config = replace(config, review_profile_path=None if profile is None else Path(profile).resolve(),
                     discover_github_provenance=False)
    _resolve_review_profile(config)
    return config


def _fetch_case(case: Case, repository: Path) -> None:
    """Fetch only frozen endpoints from the supplied mirror, without PR APIs."""
    repository.mkdir()
    mirror = case.repo.removeprefix('https://github.com/').replace('/', '_')
    for command in (['init'], ['remote', 'add', 'origin', f'https://github.com/review-bench/{mirror}.git'],
                    ['fetch', '--no-tags', 'origin', case.base, case.head], ['checkout', '--detach', case.head]):
        _run_git(repository, command, retries=0, timeout=300, error_context='frozen case materialization failed')


async def _generate(args: argparse.Namespace, plan: dict[str, Any], cases: list[Case], directory: Path) -> int:
    from daydream.benchmark.reviewbench.native import review_case

    if candidate_identity() != plan['candidate']:
        raise ValueError('executing candidate bytes differ from plan; create a fresh candidate plan')
    if args.pr is not None and not 0 <= args.pr < len(cases):
        raise ValueError('--pr is a zero-based supplied set index')
    if args.repository is not None and args.pr is None:
        raise ValueError('--repository requires a single --pr')
    config = _native_config(args.native_args)
    requested = json.loads(json.dumps(asdict(config), default=str))
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory.parent / 'configuration.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != requested:
            raise ValueError('round configuration changed; create a fresh candidate plan')
    else:
        _write_new(config_path, requested)
    (directory / 'outcomes').mkdir(exist_ok=True)
    (directory / 'findings').mkdir(exist_ok=True)
    selected = cases if args.pr is None else [cases[args.pr]]
    for case in selected:
        outcome_path = directory / 'outcomes' / f'{case.key}.json'
        candidate_path = directory / 'findings' / f'{case.key}.json'
        if outcome_path.exists() or candidate_path.exists():
            raise ValueError(f'case already has artifacts: {case.key}; create a fresh plan')
        try:
            with tempfile.TemporaryDirectory(prefix='daydream-reviewbench-') as temporary:
                repository = args.repository
                if repository is None:
                    repository = Path(temporary) / 'repository'
                    _fetch_case(case, repository)
                result = await review_case(case, repository, config, plan['producer'])
                outcome = asdict(result)
                candidate = outcome.pop('candidate')
                if outcome['status'] in {'complete', 'completed_empty'} and candidate is not None:
                    _write_new(candidate_path, candidate)
                    outcome['candidate_sha256'] = file_digest(candidate_path)
                _write_new(outcome_path, outcome)
        except (OSError, ValueError, git_ops.GitError) as exc:
            if not outcome_path.exists():
                _write_new(outcome_path, {'status': 'failed', 'diagnostics': [str(exc)]})
    accounting = _accounting(directory, cases, Path(plan['source']))
    print(json.dumps(accounting, indent=2))
    completed_outputs = set(accounting['completed']) & set(accounting['produced'])
    return 0 if all(case.key in completed_outputs for case in selected) else 1


def handle(args: argparse.Namespace) -> int:
    """Run only the explicitly selected generation or judging action."""
    try:
        if args.reviewbench_command == 'plan':
            return _plan(args)
        directory = args.directory.resolve()
        plan, cases = _load(directory)
        if args.reviewbench_command == 'report':
            from daydream.benchmark.reviewbench.evaluator import average_rounds, load_evaluation, render_report

            rounds = [_round(directory, number, plan) for number in range(1, plan['rounds'] + 1)]
            metrics = []
            complete = True
            for round_dir in rounds:
                accounting = _accounting(round_dir, cases, Path(plan['source']))
                complete = complete and accounting['complete_evaluation']
                print(json.dumps({'set': plan['set'], 'round': round_dir.name,
                                  'scope': 'complete-set' if accounting['complete_evaluation'] else 'subset',
                                  **accounting}, indent=2))
                if (round_dir / 'evaluation').exists():
                    verified = load_evaluation(Path(plan['source']), round_dir / 'findings', round_dir / 'evaluation')
                    print(render_report(Path(plan['source']), round_dir / 'evaluation' / 'results.json'))
                    metrics.append(verified['metrics'])
            if len(metrics) == 3 and plan['set'] == 'full' and complete:
                print(json.dumps(average_rounds(metrics), indent=2))
            return 0 if complete else 1
        round_dir = _round(directory, args.round, plan)
        if args.reviewbench_command == 'run':
            return anyio.run(_generate, args, plan, cases, round_dir)
        accounting = _accounting(round_dir, cases, Path(plan['source']))
        if args.reviewbench_command == 'judge':
            from daydream.benchmark.reviewbench.evaluator import evaluate_round

            evaluate_round(Path(plan['source']), round_dir / 'findings', round_dir / 'evaluation',
                           provider=args.provider, model=args.model, concurrency=args.concurrency,
                           expected_keys=accounting['produced'])
            accounting = _accounting(round_dir, cases, Path(plan['source']))
        print(json.dumps(accounting, indent=2))
        return 0 if accounting['complete_generation'] else 1
    except (OSError, ValueError, KeyError, TypeError, ProfileError, git_ops.GitError) as exc:
        print(f'ReviewBench: {exc}', file=sys.stderr)
        return 1
