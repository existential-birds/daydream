"""Operator contracts omitted by native conversion and upstream evaluator tests.

Fixtures are small synthetic metadata at the supplied revision's documented
manifest boundary; they contain no copied corpus, reference findings or scores.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from daydream import runner
from daydream.benchmark.cli import _handle_benchmark_command
from daydream.benchmark.reviewbench import cli, contract, evaluator
from daydream.benchmark.reviewbench.contract import Case, execution_accounting, load_cases, verify_source
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import commit, git, init_repo, write_and_stage
from tests.test_reviewbench_evaluator import _finding, _metrics, external_evaluator as external_evaluator
from tests.test_reviewbench_native import frozen_case


@pytest.fixture
def supplied_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / 'supplied'
    init_repo(source)
    entries = [asdict(Case('https://github.com/example/project', number, 'a' * 40, 'b' * 40))
               for number in range(1, 220)]
    write_and_stage(source, 'corpus/manifest.json', json.dumps(entries))
    write_and_stage(source, 'corpus/test/test.json', json.dumps(entries[:25]))
    for filename in ('package-lock.json', 'scripts/eval/scorer.ts', 'scripts/eval/print-summary.ts',
                     'scripts/eval/eval.ts', 'scripts/eval/matcher.ts', 'scripts/classifier/prompts.ts'):
        write_and_stage(source, filename, 'synthetic input\n')
    monkeypatch.setattr(contract, 'REVIEWBENCH_REVISION', commit(source, 'synthetic upstream contract'))
    return source


@pytest.mark.parametrize('fault', ['revision', 'modified', 'untracked'])
def test_source_pin_refuses_changed_inputs(supplied_source: Path, fault: str) -> None:
    """A fixed revision label cannot bless altered scorer/reference bytes.

    Native tests bind case Git; this owns the distinct evaluator source pin.
    """
    assert verify_source(supplied_source)['hashes']['scripts/eval/scorer.ts']
    if fault == 'revision':
        write_and_stage(supplied_source, 'new', 'other revision')
        commit(supplied_source, 'different revision')
    elif fault == 'modified':
        (supplied_source / 'scripts/eval/scorer.ts').write_text('changed metrics')
    else:
        (supplied_source / 'extra-reference.json').write_text('{}')
    with pytest.raises(ValueError, match='pinned|modified|untracked'):
        verify_source(supplied_source)


@pytest.mark.parametrize('fault', ['duplicate', 'subset', 'identity'])
def test_supplied_membership_is_exact(supplied_source: Path, fault: str) -> None:
    """Starter selection must use its membership, rather than a sorted judge limit.

    Evaluator tests accept already selected keys and cannot detect a wrong set.
    """
    assert len(load_cases(supplied_source, 'full')) == 219
    starter = load_cases(supplied_source, 'test')
    assert len(starter) == 25 and starter[-1].pr_number == 25
    path = supplied_source / 'corpus/test/test.json'
    values = json.loads(path.read_text())
    if fault == 'duplicate':
        values[-1] = values[0]
    elif fault == 'subset':
        values.pop()
    else:
        values[-1]['base'] = 'c' * 40
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match='membership|identity'):
        load_cases(supplied_source, 'test')


def test_round_accounting_keeps_absent_and_failed_work_visible() -> None:
    """Overlap-only scoring cannot imply full-corpus completion.

    Upstream metrics intentionally omit absent cases; this independent inventory
    protects failed, missing and completed-empty cases even with a valid score.
    """
    result = execution_accounting(
        ['empty', 'failed', 'absent', 'limited'],
        {'empty': {'status': 'completed_empty'}, 'failed': {'status': 'incomplete'},
         'limited': {'status': 'conversion_limited', 'conversion_omissions': [
             {'uid': 'old-side', 'reason': 'ambiguous_old_or_new_side_line_hint'},
             {'uid': 'deleted', 'reason': 'location_absent_from_head_regular_file'},
         ]}}, ['empty'], ['empty'],
    )
    assert result['completed_empty'] == ['empty']
    assert result['failed'] == ['failed', 'limited'] and result['missing'] == ['absent']
    assert result['conversion_limited'] == ['limited']
    assert result['counts'] == {'expected': 4, 'produced': 1, 'completed': 1, 'completed_empty': 1,
                                'failed': 2, 'missing': 1, 'missing_artifacts': 0, 'evaluated': 1,
                                'conversion_limited': 1, 'conversion_omitted_findings': 2}
    assert not result['complete_generation'] and not result['complete_evaluation']
    lost = execution_accounting(['lost'], {'lost': {'status': 'completed_empty'}}, [])
    assert lost['missing_artifacts'] == ['lost']
    assert not lost['complete_generation'] and not lost['complete_evaluation']
    with pytest.raises(ValueError, match='membership'):
        execution_accounting(['expected'], {}, ['foreign'])


@pytest.mark.parametrize('fault', ['prior', 'install'])
def test_new_outputs_preserve_prior_bytes_on_install_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    """Concurrent/prior output must survive failed publication.

    Existing artifact-session tests do not own benchmark plan/outcome creation.
    """
    output_directory = tmp_path / 'publication'
    output_directory.mkdir()
    path = output_directory / 'result.json'
    if fault == 'prior':
        path.write_bytes(b'valid prior evidence')
    else:
        link = os.link

        def fail(source: Any, destination: Any, **kwargs: Any) -> None:
            if Path(destination) == path:
                raise OSError('actual install fault')
            link(source, destination, **kwargs)

        monkeypatch.setattr(os, 'link', fail)
    with pytest.raises(OSError):
        cli._write_new(path, {'new': True})
    assert path.read_bytes() == b'valid prior evidence' if fault == 'prior' else not path.exists()
    assert list(output_directory.iterdir()) == ([path] if fault == 'prior' else [])


def test_operator_native_flags_use_existing_precedence(tmp_path: Path) -> None:
    """Benchmark parsing must retain repository phase settings and explicit overrides.

    Native adapter tests start with RunConfig and miss CLI/file composition.
    """
    (tmp_path / '.daydream.toml').write_text('[phases.per_stack_review]\nbackend="pi"\nmodel="phase-model"\n')
    config = cli._native_config(['--', str(tmp_path), '--backend', 'pi', '--model', 'explicit-model'])
    assert config.backend == 'pi' and config.model == 'explicit-model'
    assert config.file_config is not None
    assert config.file_config.phases['per_stack_review']['model'] == 'phase-model'
    assert config.review_profile is not None and config.dataset_capture_disabled
    with pytest.raises(ValueError, match='benchmark owns'):
        cli._native_config([str(tmp_path), '--base', 'main'])


@pytest.mark.parametrize('fault', ['plan', 'outcome', 'profile'])
def test_malformed_operator_inputs_report_failure_without_launching_review(
    supplied_source: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], fault: str,
) -> None:
    """Invalid saved JSON or review policy must produce a CLI error without a traceback.

    Profile parser tests own invalid policy content; this protects the benchmark
    command's error boundary, which previously leaked ProfileError/AttributeError.
    """
    root = tmp_path / 'runs'
    assert _handle_benchmark_command([
        'reviewbench', 'plan', str(root), '--source', str(supplied_source),
        '--candidate', 'native', '--producer', 'fixture',
    ]) == 0
    directory = root / 'native'
    capsys.readouterr()
    command = ['reviewbench', 'status', str(directory)]
    if fault == 'plan':
        (directory / 'plan.json').write_text('[]')
    elif fault == 'outcome':
        outcomes = directory / 'round-1' / 'outcomes'
        outcomes.mkdir(parents=True)
        case = load_cases(supplied_source, 'test')[0]
        (outcomes / f'{case.key}.json').write_text('[]')
    else:
        configuration = tmp_path / 'configuration'
        configuration.mkdir()
        profile = tmp_path / 'bad-profile.toml'
        profile.write_text('unknown_key = true\n')
        command = ['reviewbench', 'run', str(directory), '--', str(configuration), '--review-profile', str(profile)]
    assert _handle_benchmark_command(command) == 1
    assert 'ReviewBench:' in capsys.readouterr().err
    assert not (directory / 'round-1' / 'findings').exists()


def test_incomplete_full_round_reports_remain_visible_without_a_full_set_average(
    supplied_source: Path, external_evaluator: tuple[Path, Path, Case], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Three matching scored subsets must not masquerade as three complete full-corpus rounds.

    Evaluator tests own saved evidence; this crosses real plan/accounting/report
    files and the external-process boundary to enforce the CLI's full-set gate.
    """
    del external_evaluator  # installs only the external npm substitute; source pin remains real below
    case = load_cases(supplied_source, 'full')[0]
    write_and_stage(supplied_source, '.gitignore', 'node_modules/\nargs.json\npayload.json\n')
    write_and_stage(supplied_source, f'golden/{case.key}.json', json.dumps({'pr': asdict(case), 'findings': []}))
    monkeypatch.setattr(contract, 'REVIEWBENCH_REVISION', commit(supplied_source, 'synthetic evaluator references'))
    (supplied_source / 'payload.json').write_text(json.dumps(_metrics(case)))
    renderer = supplied_source / 'node_modules' / '.bin' / 'tsx'
    renderer.parent.mkdir(parents=True)
    renderer.write_text(f'#!{sys.executable}\nprint("supplied fixture summary")\n')
    renderer.chmod(0o755)
    root = tmp_path / 'runs'
    assert _handle_benchmark_command([
        'reviewbench', 'plan', str(root), '--source', str(supplied_source), '--candidate', 'native',
        '--producer', 'fixture', '--set', 'full', '--rounds', '3',
    ]) == 0
    directory = root / 'native'
    capsys.readouterr()
    for number in range(1, 4):
        round_dir = directory / f'round-{number}'
        candidate = round_dir / 'findings' / f'{case.key}.json'
        cli._write_new(candidate, {'pr': asdict(case), 'agent': 'fixture', 'findings': [_finding()]})
        cli._write_new(round_dir / 'outcomes' / f'{case.key}.json', {
            'status': 'complete', 'candidate_sha256': contract.file_digest(candidate),
        })
        evaluator.evaluate_round(supplied_source, candidate.parent, round_dir / 'evaluation',
                                 provider='chosen-provider', model='chosen-model')
    assert _handle_benchmark_command(['reviewbench', 'report', str(directory)]) == 1
    output = capsys.readouterr().out
    assert output.count('"scope": "subset"') == 3
    assert output.count('supplied fixture summary') == 3
    assert '"round_count"' not in output and '"scope": "complete-set"' not in output


def test_plan_run_and_status_bind_candidate_case_and_round(
    supplied_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A selected native case persists empty completion without claiming 25/219 PRs.

    This crosses CLI plan/config/run/output with real native runner and Git; native
    tests alone do not bind candidate bytes, zero-based selection or round output.
    """
    repo, case = frozen_case(tmp_path)
    # Substitute this synthetic frozen Git identity into the synthetic corpus,
    # then pin that source, rather than replacing any production loader.
    for filename in ('corpus/manifest.json', 'corpus/test/test.json'):
        path = supplied_source / filename
        entries = json.loads(path.read_text())
        entries[0] = asdict(case)
        entries[6]['pr_number'] = 1000  # preserve unique membership after replacing PR 1 with PR 7
        path.write_text(json.dumps(entries))
        git(supplied_source, 'add', filename)
    monkeypatch.setattr(contract, 'REVIEWBENCH_REVISION', commit(supplied_source, 'bind real Git fixture'))
    config_root = tmp_path / 'operator'
    config_root.mkdir()
    (config_root / '.daydream.toml').write_text('backend="pi"\nsupervisor="llm"\n')
    monkeypatch.setattr(runner, 'create_backend', lambda _name, **kwargs: EmptyReviewBackend(kwargs['cwd']))
    root = tmp_path / 'runs'
    command = ['reviewbench']
    assert _handle_benchmark_command(command + ['plan', str(root), '--source', str(supplied_source),
                                               '--candidate', 'native', '--producer', 'fixture', '--rounds', '3']) == 0
    directory = root / 'native'
    assert _handle_benchmark_command(command + ['status', str(directory), '--round', '2']) == 1
    status = json.loads(capsys.readouterr().out.split(str(directory) + '\n')[-1])
    assert len(status['missing']) == 25 and not status['produced']
    assert _handle_benchmark_command(command + ['run', str(directory), '--round', '1', '--pr', '0',
                                               '--repository', str(repo), '--', str(config_root)]) == 0
    capsys.readouterr()
    assert _handle_benchmark_command(command + ['status', str(directory), '--round', '1']) == 1
    accounting = json.loads(capsys.readouterr().out)
    assert not accounting['complete_generation'] and accounting['counts']['missing'] == 24
    candidate = directory / 'round-1/findings' / f'{case.key}.json'
    assert json.loads(candidate.read_text())['findings'] == []
    outcome = json.loads((directory / 'round-1/outcomes' / f'{case.key}.json').read_text())
    assert outcome['status'] == 'completed_empty' and outcome['terminal_result']['analysis_state'] == 'complete'
    assert not (directory / 'round-2/findings').exists()
    # Native ambient selections are part of the experiment even though they
    # never appear in asdict(RunConfig). Reject drift before creating case input.
    with monkeypatch.context() as changed:
        changed.setenv('PI_PROVIDER', 'different-provider')
        assert _handle_benchmark_command(command + ['run', str(directory), '--round', '2', '--pr', '0',
                                                   '--repository', str(repo), '--', str(config_root)]) == 1
        assert 'round configuration changed' in capsys.readouterr().err
        assert not (directory / 'round-2/findings').exists()
    (config_root / '.pi').mkdir()
    native_settings = config_root / '.pi/settings.json'
    native_settings.write_text('{"defaultProvider":"nous","defaultModel":"changed-native-model"}')
    try:
        assert _handle_benchmark_command(command + ['run', str(directory), '--round', '2', '--pr', '0',
                                                   '--repository', str(repo), '--', str(config_root)]) == 1
        assert 'round configuration changed' in capsys.readouterr().err
        assert not (directory / 'round-2/findings').exists()
    finally:
        native_settings.unlink()
    # Selecting another case against an incompatible frozen repository must
    # still fail this invocation, even though an earlier selected case succeeded.
    assert _handle_benchmark_command(command + ['run', str(directory), '--round', '1', '--pr', '1',
                                               '--repository', str(repo), '--', str(config_root)]) == 1
    failed_case = load_cases(supplied_source, 'test')[1]
    failed = json.loads((directory / 'round-1/outcomes' / f'{failed_case.key}.json').read_text())
    assert failed['status'] == 'failed'
    assert not (directory / 'round-1/findings' / f'{failed_case.key}.json').exists()
    capsys.readouterr()
    # Corrupting paired candidate identity must refuse scoring/status, preserving
    # the original outcome and all other rounds.
    raw = json.loads(candidate.read_text())
    raw['pr']['base'] = 'c' * 40
    candidate.write_text(json.dumps(raw))
    assert _handle_benchmark_command(command + ['status', str(directory)]) == 1
    assert 'unpaired, changed or mismatched' in capsys.readouterr().err
    assert not fake_gh.calls('POST')
