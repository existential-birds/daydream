"""Real Git/gh setup tests; only the live browser exchange is replaced."""
from pathlib import Path
from typing import Any

import pytest

from daydream import bot_manifest, bot_setup, config, git_ops
from daydream.github_app import APP_ID_ENV, APP_PRIVATE_KEY_ENV, AppCredentials, GitHubAppError
from daydream.templates import workflow_template_files
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import commit as _commit, git as _git
from tests.harness.rsa import generate_rsa_pem
from tests.harness.scripts import cli_main


def _install_workflows(
    repo: Path, *, drift: tuple[str, str, str] | None = None, unique: bool = True, message: str = "add workflows",
    push: bool = True,
) -> Path:
    """Commit and optionally push packaged workflows, with one anchor-checked drift.

    ``unique=False`` permits a repeated token but still replaces only its first occurrence.
    """
    workflows_dir = repo / ".github/workflows"
    workflows_dir.mkdir(parents=True)
    for template in workflow_template_files():
        content = template.read_text()
        if drift is not None and template.name == drift[0]:
            if unique:
                assert content.count(drift[1]) == 1, f"{template.name}: anchor no longer present"
            else:
                assert drift[1] in content, f"{template.name}: anchor no longer present"
            content = content.replace(drift[1], drift[2], 1)
        (workflows_dir / template.name).write_text(content)
    _git(repo, "add", ".github/workflows")
    _commit(repo, message)
    if push:
        _git(repo, "push", "origin", "main")
    return workflows_dir


def test_app_manifest_requests_issue_write_for_improve_publication() -> None:
    manifest = bot_manifest._manifest_payload(redirect_url="http://localhost:8080/callback",)
    assert manifest["default_events"] == ["issue_comment", "workflow_run"]

    permissions = manifest["default_permissions"]
    assert isinstance(permissions, dict)
    assert permissions["issues"] == "write"

def test_callback_listener_exchanges_code_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_exchange(repo: Path, code: str, **_kwargs: Any) -> tuple[AppCredentials, str]:
        captured.update(repo=repo, code=code)
        return AppCredentials(7, "-----BEGIN-----\n"), "acme-bot"

    monkeypatch.setattr("daydream.bot_manifest.exchange_manifest_code", fake_exchange)
    listener = bot_manifest._ManifestListener(repo_dir=Path("."), org=None)
    creds, actual_slug = listener._handle_code("codeXYZ")
    assert captured == {"repo": Path("."), "code": "codeXYZ"}
    assert creds.app_id == 7
    assert creds.private_key == "-----BEGIN-----\n"
    assert actual_slug == "acme-bot"

def test_missing_code_raises_cancelled_and_never_exchanges(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty/missing callback code (user declined) aborts with a clear error."""
    called = False

    def fake_exchange(repo: Any, code: Any, **_kwargs: Any) -> tuple[Any, ...]:
        nonlocal called
        called = True
        return AppCredentials(1, "pem"), "slug"

    monkeypatch.setattr("daydream.bot_manifest.exchange_manifest_code", fake_exchange)
    listener = bot_manifest._ManifestListener(repo_dir=Path("."), org=None)
    with pytest.raises(GitHubAppError, match="App registration was cancelled"):
        listener._handle_code("")
    with pytest.raises(GitHubAppError, match="App registration was cancelled"):
        listener._handle_code(None)
    assert called is False




def test_land_workflows_idempotent_returns_sentinel_when_all_present(fake_gh: FakeGh, repo_with_origin: Path) -> None:
    """Matching templates produce a no-op sentinel distinct from a PR URL, without creating either."""
    wf = repo_with_origin / ".github/workflows"
    wf.mkdir(parents=True)

    for template in workflow_template_files():
        (wf / template.name).write_text(template.read_text())

    default = git_ops.default_branch(repo_with_origin)
    result = bot_setup.land_workflows(repo_with_origin, branch="daydream/setup-bot")

    assert result == bot_setup.WORKFLOWS_ALREADY_INSTALLED
    # Sentinel must be distinguishable from a PR URL; no branch/PR side effects.
    assert not result.startswith("http")
    assert git_ops.current_branch(repo_with_origin) == default
    assert not git_ops.ref_exists(repo_with_origin, "origin/daydream/setup-bot")




def test_verify_healthy_install_passes_all_checks(
    fake_gh: FakeGh, repo_with_origin: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The real doctor reports each incomplete installation stage until external repair."""
    pem = generate_rsa_pem()
    monkeypatch.setattr(bot_setup, "run_setup", lambda *a, **kw: pytest.fail("--verify must remain read-only"))

    fake_gh.serve_secret_list(["DAYDREAM_APP_ID", "ANTHROPIC_API_KEY"])
    fake_gh.serve_variable_list([])
    fake_gh.serve_installations([{"account": {"login": "o"}}])
    fake_gh.set_response("GET", "/app", value={"permissions": dict(config.APP_PERMISSIONS), "slug": "acme-bot"})

    assert cli_main(["setup", str(repo_with_origin), "--repo", "o/r", "--verify"]) == 1
    incomplete = " ".join(capsys.readouterr().out.replace("║", " ").split())
    assert "Missing secret(s): DAYDREAM_APP_PRIVATE_KEY" in incomplete
    assert "gh secret set <NAME> --repo o/r" in incomplete
    assert "Missing variable DAYDREAM_BOT_HANDLE" in incomplete
    assert "gh variable set DAYDREAM_BOT_HANDLE --repo o/r" in incomplete
    assert "Skipped: no local App credentials" in incomplete
    assert "Missing workflow file(s):" in incomplete
    assert all(name in incomplete for name in ("daydream-review.yml", "daydream-command.yml", "daydream-post.yml"))
    assert "Out of date workflow file(s)" not in incomplete

    _install_workflows(repo_with_origin, message="add workflows")
    fake_gh.serve_variable_list([config.BOT_HANDLE_VAR])
    assert cli_main(["setup", str(repo_with_origin), "--repo", "o/r", "--verify"]) == 1
    remaining = " ".join(capsys.readouterr().out.replace("║", " ").split())
    assert "Missing secret(s): DAYDREAM_APP_PRIVATE_KEY" in remaining
    assert "Missing variable" not in remaining
    assert "Missing workflow file(s)" not in remaining

    fake_gh.serve_secret_list(list(config.SETUP_SECRET_NAMES))
    monkeypatch.setenv(APP_ID_ENV, "7")
    monkeypatch.setenv(APP_PRIVATE_KEY_ENV, pem)
    result = bot_setup.run_verify(repo_with_origin, scope=bot_setup.Scope(repo="o/r"))
    assert result.ok is True
    assert all(c.passed for c in result.checks)
    assert fake_gh.calls("GET", "/app/installations")
    assert fake_gh.secret_set_calls() == []
    assert fake_gh.variable_set_calls() == []

@pytest.mark.parametrize(("target", "old", "new"),
    [pytest.param("daydream-review.yml",
            "on:\n  workflow_dispatch:",
            "on:\n  pull_request:\n    types: [opened, ready_for_review]\n  workflow_dispatch:",
            id="unapproved-pull-request-trigger",
        ), pytest.param("daydream-review.yml",
            "        type: choice\n        options: [review, sequence, flowchart]\n",
            "        type: string\n",
            id="unbounded-command-input",
        ), pytest.param("daydream-command.yml",
            '            -f approved_head_sha="$HEAD_SHA" \\\n',
            "", id="head-binding-removed",
        ),
    ],
)
def test_verify_rejects_workflow_that_loosens_the_command_contract(
    fake_gh: FakeGh, repo_with_origin: Path, target: str, old: str, new: str
) -> None:
    """Reject automatic PR review, an unbounded command selector, and missing approved-head binding."""

    _install_workflows(repo_with_origin, drift=(target, old, new), message="add loosened workflows")

    result = bot_setup.run_verify(repo_with_origin, scope=bot_setup.Scope(repo="o/r"))
    workflows_check = next(check for check in result.checks if check.name == "workflows")
    assert result.ok is False
    assert workflows_check.passed is False
    assert f"Out of date workflow file(s): .github/workflows/{target}" in workflows_check.detail
    assert "Missing workflow file(s)" not in workflows_check.detail

def test_verify_accepts_customized_workflow_with_intact_gate(fake_gh: FakeGh, repo_with_origin: Path) -> None:
    """A divergent-but-gated workflow (e.g. a different backend) passes with a warning."""
    fake_gh.serve_secret_list(list(config.SETUP_SECRET_NAMES))
    fake_gh.serve_variable_list([config.BOT_HANDLE_VAR])

    _install_workflows(repo_with_origin,
        # Backend-variant customization: the gate (approved_head_sha input,
        # no pull_request trigger) is intact, credential/backend swapped.
        drift=("daydream-review.yml", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"), unique=False,
        message="add customized workflows",
    )

    result = bot_setup.run_verify(repo_with_origin, scope=bot_setup.Scope(repo="o/r"))
    workflows_check = next(check for check in result.checks if check.name == "workflows")
    assert result.ok is True
    assert workflows_check.passed is True
    assert "daydream-review.yml" in workflows_check.detail
    assert "intentionally unsupported" in workflows_check.detail

def test_land_workflows_warns_before_overwriting_customized_workflow(
    fake_gh: FakeGh, repo_with_origin: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """land_workflows warns that customization is unsupported before replacing it."""

    workflows_dir = _install_workflows(
        repo_with_origin, drift=("daydream-review.yml", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"), unique=False,
        message="add customized workflows", push=False,
    )

    warnings: list[str] = []
    monkeypatch.setattr("daydream.bot_setup.print_warning", lambda console, msg: warnings.append(msg))
    fake_gh.set_response("pr-create", value="https://github.com/o/r/pull/5")
    fake_gh.set_response("pr-list", value=[])

    review_template = next(t for t in workflow_template_files() if t.name == "daydream-review.yml")
    url = bot_setup.land_workflows(repo_with_origin, branch="daydream/setup-bot")

    assert url != bot_setup.WORKFLOWS_ALREADY_INSTALLED
    assert any("daydream-review.yml" in w and "intentionally unsupported" in w for w in warnings)
    # The customization was replaced by the packaged template on the branch.
    assert (workflows_dir / "daydream-review.yml").read_text() == review_template.read_text()





def test_setup_verb_full_auto_deposits_secrets_and_opens_pr(
    fake_gh: FakeGh, repo_with_origin: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Run the setup CLI with real Git/gh and a minted PEM; replace only the browser exchange."""
    pem = generate_rsa_pem()
    monkeypatch.setattr(
        "daydream.bot_setup.register_app_via_manifest", lambda repo, org=None: (AppCredentials(7, pem), "acme-bot"),
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    # App already installed on the owner → the Install-click wait is satisfied.
    fake_gh.serve_installations([{"account": {"login": "o"}}])
    fake_gh.set_response("pr-list", value=[])
    fake_gh.set_response("pr-create", value="https://github.com/o/r/pull/5")

    code = cli_main(["setup", str(repo_with_origin), "--repo", "o/r"])

    assert code == 0
    assert {c.name for c in fake_gh.secret_set_calls()} == set(config.SETUP_SECRET_NAMES)
    variable = fake_gh.variable_set_calls()[-1]
    assert variable.name == config.BOT_HANDLE_VAR
    assert variable.argv[variable.argv.index("--body") + 1] == "acme-bot"
    wf = repo_with_origin / ".github/workflows"
    assert {p.name for p in wf.glob("*.yml")} == {"daydream-review.yml", "daydream-command.yml", "daydream-post.yml"}
    # The bot goes live on merge: a reviewable PR was opened on a non-default branch.
    pr_calls = fake_gh.command_calls("pr create")
    assert len(pr_calls) == 1
    assert git_ops.ref_exists(repo_with_origin, "origin/daydream/setup-bot")
    assert git_ops.current_branch(repo_with_origin) != git_ops.default_branch(repo_with_origin)
    assert "https://github.com/o/r/pull/5" in capsys.readouterr().out

def test_land_workflows_pr_lookup_failure_never_creates_duplicate_pr(fake_gh: FakeGh, repo_with_origin: Path) -> None:
    fake_gh.set_response("pr-list", value={"__error__": "authentication required"})
    fake_gh.set_response("pr-create", value="https://github.com/o/r/pull/unsafe")

    with pytest.raises(git_ops.GitError, match="authentication required"):
        bot_setup.land_workflows(repo_with_origin, branch="daydream/setup-bot")

    assert fake_gh.command_calls("pr create") == []

def test_setup_fails_cleanly_when_key_absent_and_noninteractive(
    fake_gh: FakeGh, repo_with_origin: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing key with non-TTY input must fail before registration or secret deposit."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("daydream.bot_setup.sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("daydream.bot_setup.getpass.getpass", lambda *_: pytest.fail("non-TTY setup must not prompt"))
    register_called = False

    def _fail_register(repo: Path, org: str | None = None) -> tuple[AppCredentials, str]:
        nonlocal register_called
        register_called = True
        raise AssertionError("registration must not run before the key pre-flight passes")

    monkeypatch.setattr("daydream.bot_setup.register_app_via_manifest", _fail_register)

    code = cli_main(["setup", str(repo_with_origin), "--repo", "o/r"])

    assert code == 1
    assert register_called is False
    assert fake_gh.secret_set_calls() == []


def test_prompt_for_anthropic_key_reads_hidden_input_on_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a TTY the seam returns the hidden-entered key, stripped."""
    monkeypatch.setattr("daydream.bot_setup.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("daydream.bot_setup.getpass.getpass", lambda prompt="": "  sk-ant-typed  ")
    assert bot_setup._prompt_for_anthropic_key() == "sk-ant-typed"




@pytest.mark.parametrize(("scope", "expected"), [
    (bot_setup.Scope(repo="owner/repo"), "owner"),
    (bot_setup.Scope(org="acme-org"), "acme-org"),
])
def test_bot_handle_falls_back_to_scope_owner(scope: bot_setup.Scope, expected: str) -> None:
    assert bot_setup._bot_handle_for(None, scope) == expected
