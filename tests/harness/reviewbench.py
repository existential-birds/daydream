"""Frozen ReviewBench inputs and actual reviewer-process test support."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import shlex
import signal
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from daydream import git_ops
from daydream.benchmark.reviewbench.contract import PUBLIC_PROFILE_SCHEMA, REVIEWBENCH_REVISION, Case, verify_source
from daydream.benchmark.reviewbench.native import freeze_reviewer
from tests.deep_orchestrator.empty_synthesis_support import empty_review_config
from tests.harness.git_helpers import commit, git, init_repo, write_and_stage


@pytest.fixture
def supplied_source() -> Path:
    configured = os.environ.get("DAYDREAM_REVIEWBENCH_SOURCE")
    if not configured:
        pytest.skip("set DAYDREAM_REVIEWBENCH_SOURCE to the pinned checkout with npm ci installed")
    source = Path(configured).resolve()
    verify_source(source)
    if not (source / "node_modules" / ".bin" / "tsx").is_file():
        pytest.fail("Run npm ci in the explicitly selected ReviewBench source checkout")
    return source


def frozen_case(tmp_path: Path, *, additional_sources: int = 0, divergent_base: bool = False) -> tuple[Path, Case]:
    repo = tmp_path / "frozen-input"
    init_repo(repo)
    write_and_stage(repo, "README.md", "Synthetic native ReviewBench integration fixture.\n")
    base = commit(repo, "base")
    git(repo, "checkout", "-b", "candidate")
    write_and_stage(repo, "api.py", "def resolve(value):\n    return value[0]\n")
    for index in range(additional_sources):
        write_and_stage(repo, f"source_{index}.py", f"VALUE = {index}\n")
    head = commit(repo, "candidate")
    if divergent_base:
        git(repo, 'checkout', '--detach', base)
        write_and_stage(repo, 'base-progress.txt', 'The upstream branch advanced independently.\n')
        base = commit(repo, 'advanced base tip')
        git(repo, 'checkout', '--detach', head)
        assert base not in git(repo, 'for-each-ref', '--format=%(objectname)', 'refs/heads')
    git_ops.update_refs(repo, {'refs/reviewbench/base': base, 'refs/reviewbench/head': head})
    return repo, Case("https://github.com/example/project", 7, base, head)


def finding(**overrides: Any) -> dict[str, Any]:
    return {"id": 1, "file": "api.py", "line": 2, "description": "Empty input raises IndexError",
            "rationale": "The API accepts an empty sequence.", "evidence": "api.py:2 indexes value[0].",
            "severity": "medium", "confidence": "MEDIUM", **overrides}


def public_input(tmp_path: Path, *, additional_sources: int = 0, divergent_base: bool = False) -> dict[str, str]:
    repo, case = frozen_case(tmp_path, additional_sources=additional_sources, divergent_base=divergent_base)
    inputs = tmp_path / 'pr'
    inputs.mkdir()
    pr = {**asdict(case), 'title': 'Frozen supplied title', 'body': 'Frozen supplied author explanation'}
    (inputs / 'pr.json').write_text(json.dumps(pr))
    (inputs / 'diff.patch').write_text(git_ops.diff_paths(repo, case.base, case.head, ['.'], merge_base_diff=True))
    profile = {'schema': PUBLIC_PROFILE_SCHEMA, 'upstream_revision': REVIEWBENCH_REVISION, 'pr': pr,
               'task_id': 7, 'evaluation_id': 99,
               'reviewer': freeze_reviewer(empty_review_config(repo, tmp_path / 'ignored', backend='codex',
                   model='gpt-6.1-sol', reasoning_effort='high')),
               'candidate': {'wheel_name': 'daydream-0.30.0-py3-none-any.whl', 'wheel_sha256': 'b' * 64}}
    profile['profile_sha256'] = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(',', ':'))
                                               .encode()).hexdigest()
    (inputs / 'daydream-profile.json').write_text(json.dumps(profile))
    canonical = tmp_path / 'subscription'
    canonical.mkdir()
    payload = base64.urlsafe_b64encode(json.dumps({'exp': 9000000000}).encode()).decode().rstrip('=')
    (canonical / 'auth.json').write_text(json.dumps({'auth_mode': 'chatgpt', 'tokens': {
        'access_token': f'header.{payload}.signature', 'refresh_token': 'private-refresh-canary',
        'id_token': 'private-id-canary', 'account_id': 'test-account'}}))
    return {'DAYDREAM_RB_REPOSITORY': str(repo), 'DAYDREAM_RB_INPUTS': str(inputs),
            'DAYDREAM_RB_OUTPUT': str(tmp_path / 'out/findings.json'),
            'DAYDREAM_RB_EVIDENCE': str(tmp_path / 'logs/daydream'),
            'DAYDREAM_RB_CODEX_HOME': str(canonical), 'PATH': os.environ['PATH'], 'HOME': str(tmp_path),
            'OPENAI_API_KEY': 'forbidden-api-canary', 'GH_TOKEN': 'forbidden-github-canary',
            'OPENAI_BASE_URL': 'https://unselected.invalid'}


class LocalReviewerEnvironment:
    """Buffered exec with Harbor's cancellation boundary: children outlive awaits."""

    def __init__(self, environment: dict[str, str], repository: Path, *,
                 publication_fault: str | None = None, close_stdout: bool = False, foreign_stderr: bytes = b'',
                 entrypoint_bootstrap: str | None = None) -> None:
        self.environment = environment
        self.repository = repository
        self.publication_fault = publication_fault
        self.close_stdout = close_stdout
        self.foreign_stderr = foreign_stderr
        self.entrypoint_bootstrap = entrypoint_bootstrap
        self.results: list[SimpleNamespace] = []
        self.children: list[asyncio.subprocess.Process] = []
        self.collectors: list[asyncio.Task[tuple[bytes, bytes]]] = []

    async def exec(self, command: str, **kwargs: Any) -> Any:
        arguments = shlex.split(command)
        assert arguments[0] == 'python'
        arguments[0] = sys.executable
        inject = self.publication_fault is not None or self.foreign_stderr or self.entrypoint_bootstrap
        if inject and arguments[1:4] == [
            '-I', '-m', 'daydream.benchmark.reviewbench.harbor_process',
        ]:
            # Patch the actual filesystem install only in the supervisor. The
            # production entrypoint/native CLI remain separate, unpatched children.
            bootstrap = ('import errno,os,runpy,sys\nfrom pathlib import Path\n' +
                f'os.write(2,{self.foreign_stderr!r})\n' +
                'original_link = os.link\n'
                'def fail_link(source,destination,**kwargs):\n'
                f'    if Path(destination).name == {self.publication_fault!r}:\n'
                '        raise OSError(errno.ENOSPC,"publication-secret-canary")\n'
                '    return original_link(source,destination,**kwargs)\n'
                'os.link = fail_link\n'
                + ('import asyncio\noriginal_spawn=asyncio.create_subprocess_exec\n'
                   'async def spawn(*args,**kwargs):\n'
                   '    if args[1:4] == ("-I","-m","daydream.benchmark.reviewbench.harbor_entrypoint"):\n'
                   f'        args=(args[0],"-I","-c",{self.entrypoint_bootstrap!r})\n'
                   '    return await original_spawn(*args,**kwargs)\n'
                   'asyncio.create_subprocess_exec=spawn\n' if self.entrypoint_bootstrap else '') +
                'sys.argv = ["daydream.benchmark.reviewbench.harbor_process", *sys.argv[1:]]\n'
                'runpy.run_module(sys.argv[0],run_name="__main__",alter_sys=True)\n')
            arguments = [sys.executable, '-I', '-c', bootstrap, *arguments[4:]]
        writer: int | None = None
        if self.close_stdout:
            reader, writer = os.pipe()
            os.close(reader)  # A real collector pipe with no remaining reader.
        try:
            child = await asyncio.create_subprocess_exec(*arguments, cwd=self.repository, env=self.environment,
                stdout=writer if writer is not None else asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, start_new_session=True)
        finally:
            if writer is not None:
                os.close(writer)
        self.children.append(child)
        collector = asyncio.create_task(child.communicate())
        self.collectors.append(collector)
        stdout, stderr = await asyncio.shield(collector)
        result = SimpleNamespace(return_code=child.returncode, stdout=(stdout or b'').decode(), stderr=stderr.decode())
        self.results.append(result)
        return result

    async def close(self) -> None:
        for child in self.children:
            if child.returncode is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        await asyncio.wait_for(asyncio.gather(*self.collectors, return_exceptions=True), timeout=5)
