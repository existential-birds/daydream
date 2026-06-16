"""Real-path tests for :mod:`daydream.bot_setup`.

Task 5 covers the App-from-manifest localhost registration seam. The live
browser/manifest leg (binding a port, opening a browser, blocking on the
callback) is isolated behind :class:`daydream.bot_setup._ManifestListener` so
the code-exchange behavior is testable without real GitHub: the test drives
``_handle_code`` directly, monkeypatching only the manifest-code exchange.
"""

import json
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from daydream import bot_setup, cli, config, git_ops
from daydream.github_app import APP_ID_ENV, APP_PRIVATE_KEY_ENV, AppCredentials, GitHubAppError
from tests.harness.fake_gh import FakeGh, install_fake_gh


def _real_pem() -> str:
    """Generate a real RSA PEM so ``mint_jwt`` can sign a valid App JWT."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


@pytest.fixture
def fake_gh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeGh:
    """Install the fake ``gh`` binary on PATH for real-path subprocess tests."""
    return install_fake_gh(tmp_path / "fake-gh-bin", monkeypatch)


def test_callback_listener_captures_code_then_exchanges(monkeypatch):
    """The callback seam exchanges the manifest code for creds + slug."""
    monkeypatch.setattr(
        "daydream.bot_setup.exchange_manifest_code",
        lambda repo, code: (AppCredentials(7, "-----BEGIN-----\n"), "acme-bot"),
    )
    listener = bot_setup._ManifestListener(repo_dir=Path("."), org=None)
    creds, slug = listener._handle_code("codeXYZ")
    assert creds.app_id == 7
    assert creds.private_key == "-----BEGIN-----\n"
    assert slug == "acme-bot"


def test_callback_listener_passes_repo_dir_and_code_through(monkeypatch):
    """The seam threads the listener's repo_dir and the callback code unchanged."""
    captured: dict[str, object] = {}

    def fake_exchange(repo, code):
        captured["repo"] = repo
        captured["code"] = code
        return AppCredentials(42, "pem"), "slug-x"

    monkeypatch.setattr("daydream.bot_setup.exchange_manifest_code", fake_exchange)
    listener = bot_setup._ManifestListener(repo_dir=Path("/tmp/repo"), org="acme")
    creds, slug = listener._handle_code("codeABC")
    assert captured["repo"] == Path("/tmp/repo")
    assert captured["code"] == "codeABC"
    assert creds.app_id == 42 and slug == "slug-x"


def test_missing_code_raises_cancelled_and_never_exchanges(monkeypatch):
    """An empty/missing callback code (user declined) aborts with a clear error."""
    called = False

    def fake_exchange(repo, code):
        nonlocal called
        called = True
        return AppCredentials(1, "pem"), "slug"

    monkeypatch.setattr("daydream.bot_setup.exchange_manifest_code", fake_exchange)
    listener = bot_setup._ManifestListener(repo_dir=Path("."), org=None)
    with pytest.raises(GitHubAppError, match="App registration was cancelled"):
        listener._handle_code("")
    with pytest.raises(GitHubAppError, match="App registration was cancelled"):
        listener._handle_code(None)
    assert called is False


# --- Task 6: deposit_secrets ------------------------------------------------


def test_deposit_sets_three_secrets_and_handle_var_with_pem_off_argv(
    fake_gh: FakeGh, git_repo: Path
) -> None:
    """deposit_secrets sets the three canonical secrets + handle var, PEM off argv."""
    creds = AppCredentials(7, "-----BEGIN RSA PRIVATE KEY-----\nx\n")
    bot_setup.deposit_secrets(
        git_repo,
        creds,
        anthropic_key="sk-ant",
        bot_handle="acme-bot",
        scope=bot_setup.Scope(repo="o/r"),
    )
    set_names = {c.name for c in fake_gh.secret_set_calls()}
    assert set_names == set(config.SETUP_SECRET_NAMES)
    assert fake_gh.variable_set_calls()[-1].name == config.BOT_HANDLE_VAR
    assert all("BEGIN RSA" not in " ".join(c.argv) for c in fake_gh.secret_set_calls())


# --- Task 7: land_workflows -------------------------------------------------


def test_land_workflows_writes_three_files_on_branch_and_opens_pr(
    fake_gh: FakeGh, repo_with_origin: Path
) -> None:
    """land_workflows copies the templates on a new branch, pushes, opens a PR."""
    fake_gh.set_response("pr-create", value="https://github.com/o/r/pull/3")
    url = bot_setup.land_workflows(repo_with_origin, branch="daydream/setup-bot")
    wf = repo_with_origin / ".github/workflows"
    assert {p.name for p in wf.glob("*.yml")} == {
        "daydream-review.yml",
        "daydream-command.yml",
        "daydream-post.yml",
    }
    assert git_ops.ref_exists(repo_with_origin, "origin/daydream/setup-bot")
    assert git_ops.current_branch(repo_with_origin) != git_ops.default_branch(repo_with_origin)
    assert url == "https://github.com/o/r/pull/3"


def test_land_workflows_idempotent_returns_sentinel_when_all_present(
    fake_gh: FakeGh, repo_with_origin: Path
) -> None:
    """If all three templates already exist verbatim, skip branch/PR and signal it.

    The "already installed" sentinel must be distinguishable from a PR URL so
    the caller can surface the no-op without parsing a fake URL.
    """
    wf = repo_with_origin / ".github/workflows"
    wf.mkdir(parents=True)
    from daydream.templates import workflow_template_files

    for template in workflow_template_files():
        (wf / template.name).write_text(template.read_text())

    default = git_ops.default_branch(repo_with_origin)
    result = bot_setup.land_workflows(repo_with_origin, branch="daydream/setup-bot")

    assert result == bot_setup.WORKFLOWS_ALREADY_INSTALLED
    # The sentinel must be distinguishable from any PR URL the caller surfaces.
    assert not result.startswith("http")
    # No branch/PR side effects: still on the default branch, branch not pushed.
    assert git_ops.current_branch(repo_with_origin) == default
    assert not git_ops.ref_exists(repo_with_origin, "origin/daydream/setup-bot")


# --- Task 8: run_verify (the doctor) ----------------------------------------


def test_verify_reports_missing_secret_with_remediation(fake_gh: FakeGh, git_repo: Path) -> None:
    """N=1 missing secret → ok is False and the failed check names it + remediation."""
    fake_gh.serve_secret_list(["DAYDREAM_APP_ID", "ANTHROPIC_API_KEY"])  # PRIVATE_KEY absent
    fake_gh.serve_variable_list(["DAYDREAM_BOT_HANDLE"])
    fake_gh.serve_installations([{"account": {"login": "o"}}])
    result = bot_setup.run_verify(git_repo, scope=bot_setup.Scope(repo="o/r"))
    assert result.ok is False
    failed = [c for c in result.checks if not c.passed]
    assert any("DAYDREAM_APP_PRIVATE_KEY" in c.detail for c in failed)


def test_verify_healthy_install_passes_all_checks(
    fake_gh: FakeGh, repo_with_origin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete install (creds + secrets + var + installed App + workflows) → ok is True.

    Exercises every required check on its passing shape, and — with local App
    creds present — drives the App-installed (``/app/installations``) and
    permission (``GET /app``) checks through the real ``gh`` subprocess seam.

    Uses ``repo_with_origin`` (has a bare remote) so the workflows check can
    resolve ``origin/main`` via ``git show origin/main:.github/workflows/…``.
    The workflows are committed to local ``main`` and pushed to origin so the
    check passes.
    """
    from tests.conftest import _commit, _git

    pem = _real_pem()
    monkeypatch.setenv(APP_ID_ENV, "7")
    monkeypatch.setenv(APP_PRIVATE_KEY_ENV, pem)

    # All three secrets + the handle variable deposited.
    fake_gh.serve_secret_list(list(config.SETUP_SECRET_NAMES))
    fake_gh.serve_variable_list([config.BOT_HANDLE_VAR])
    # The App is installed on the target owner.
    fake_gh.serve_installations([{"account": {"login": "o"}}])
    # The App grants at least the required permissions.
    fake_gh.set_response("GET", "/app", value={"permissions": dict(config.APP_PERMISSIONS), "slug": "acme-bot"})

    # The three workflow files exist on the default branch.
    # _check_workflows resolves them via `git show origin/<base>:<path>`, so
    # the commit must be pushed to the bare remote (origin).
    wf = repo_with_origin / ".github/workflows"
    wf.mkdir(parents=True)
    from daydream.templates import workflow_template_files

    for template in workflow_template_files():
        (wf / template.name).write_text(template.read_text())
    _git(repo_with_origin, "add", ".github/workflows")
    _commit(repo_with_origin, "add workflows")
    _git(repo_with_origin, "push", "origin", "main")

    result = bot_setup.run_verify(repo_with_origin, scope=bot_setup.Scope(repo="o/r"))
    assert result.ok is True
    assert all(c.passed for c in result.checks)
    # The App-installed check actually consulted the installations endpoint.
    assert fake_gh.calls("GET", "/app/installations")


# --- Task 9: CLI `setup` verb + run_setup orchestrator ----------------------


def cli_main(argv: list[str]) -> int:
    """Drive ``cli.main`` with ``argv`` (the production entrypoint) and return its exit code."""
    saved = sys.argv
    sys.argv = ["daydream", *argv]
    try:
        cli.main()
    except SystemExit as exc:  # main() always exits via sys.exit
        return int(exc.code or 0)
    finally:
        sys.argv = saved
    raise AssertionError("cli.main() must exit via sys.exit")


def _pr_create_calls(fake_gh: FakeGh) -> list[dict]:
    """Read the recorded ``gh pr create`` invocations from the shim's call log."""
    path = fake_gh.bin_dir / "calls.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("kind") == "pr create"
    ]


def test_setup_verb_full_auto_deposits_secrets_and_opens_pr(
    fake_gh: FakeGh, repo_with_origin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end ``daydream setup`` (full-auto) through cli.main: secrets land, PR opens.

    Mocks ONLY the live browser/manifest seam. The App-from-manifest leg
    returns freshly minted creds with a real PEM so the install re-check can
    mint a valid App JWT; the fake ``gh`` shows the App already installed on
    the target owner, so the manual-Install wait is auto-satisfied (no stdin
    block). Asserts observable outcomes: exit 0, the three canonical secrets
    set, and a PR created (the bot goes live on merge).
    """
    pem = _real_pem()
    monkeypatch.setattr(
        "daydream.bot_setup.register_app_via_manifest",
        lambda repo, org=None: (AppCredentials(7, pem), "acme-bot"),
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    # App already installed on the owner → the Install-click wait is satisfied.
    fake_gh.serve_installations([{"account": {"login": "o"}}])
    fake_gh.set_response("pr-create", value="https://github.com/o/r/pull/5")

    code = cli_main(["setup", str(repo_with_origin), "--repo", "o/r"])

    assert code == 0
    assert {c.name for c in fake_gh.secret_set_calls()} == set(config.SETUP_SECRET_NAMES)
    assert fake_gh.variable_set_calls()[-1].name == config.BOT_HANDLE_VAR
    # The bot goes live on merge: a reviewable PR was opened on a non-default branch.
    pr_calls = _pr_create_calls(fake_gh)
    assert len(pr_calls) == 1
    assert git_ops.ref_exists(repo_with_origin, "origin/daydream/setup-bot")


def test_setup_prompts_for_anthropic_key_when_absent_and_deposits_it(
    fake_gh: FakeGh, repo_with_origin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``daydream setup`` with no ``ANTHROPIC_API_KEY`` in the env prompts for it.

    A new operator should not be stranded on a pre-flight error: when the key
    is absent, setup prompts (hidden input) and deposits what the operator
    enters. Mocks the manifest seam and the prompt seam only; asserts the
    observable outcome — the ``ANTHROPIC_API_KEY`` secret is set with the
    prompted value (read off the recorded stdin, never argv) and setup exits 0.
    """
    pem = _real_pem()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "daydream.bot_setup.register_app_via_manifest",
        lambda repo, org=None: (AppCredentials(7, pem), "acme-bot"),
    )
    monkeypatch.setattr("daydream.bot_setup._prompt_for_anthropic_key", lambda: "sk-ant-prompted")
    fake_gh.serve_installations([{"account": {"login": "o"}}])
    fake_gh.set_response("pr-create", value="https://github.com/o/r/pull/6")

    code = cli_main(["setup", str(repo_with_origin), "--repo", "o/r"])

    assert code == 0
    key_calls = [c for c in fake_gh.secret_set_calls() if c.name == "ANTHROPIC_API_KEY"]
    assert len(key_calls) == 1
    assert key_calls[0].stdin.strip() == "sk-ant-prompted"
    assert "sk-ant-prompted" not in " ".join(key_calls[0].argv)


def test_setup_fails_cleanly_when_key_absent_and_noninteractive(
    fake_gh: FakeGh, repo_with_origin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No key + a non-interactive stdin → clean pre-flight exit 1, nothing deposited.

    Unattended/CI runs must not block on a prompt; the prompt seam returns
    ``None`` for a non-TTY stdin so the orchestrator surfaces the pre-flight
    error and never registers an App or sets a secret.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("daydream.bot_setup._prompt_for_anthropic_key", lambda: None)
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


def test_prompt_for_anthropic_key_returns_none_on_non_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    """The prompt seam returns None (never blocks) when stdin is not a TTY."""
    monkeypatch.setattr("daydream.bot_setup.sys.stdin.isatty", lambda: False)
    assert bot_setup._prompt_for_anthropic_key() is None


def test_prompt_for_anthropic_key_reads_hidden_input_on_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a TTY the seam returns the hidden-entered key, stripped."""
    monkeypatch.setattr("daydream.bot_setup.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("daydream.bot_setup.getpass.getpass", lambda prompt="": "  sk-ant-typed  ")
    assert bot_setup._prompt_for_anthropic_key() == "sk-ant-typed"


def test_setup_verify_flag_exits_nonzero_when_incomplete(
    fake_gh: FakeGh, git_repo: Path
) -> None:
    """``daydream setup <dir> --repo o/r --verify`` on an incomplete install exits 1.

    No secrets, no variable → the doctor's required checks fail → the handler
    surfaces a non-zero exit and never registers an App or sets a secret.
    """
    fake_gh.serve_secret_list([])
    fake_gh.serve_variable_list([])

    assert cli_main(["setup", str(git_repo), "--repo", "o/r", "--verify"]) == 1
    # Read-only doctor: nothing was deposited.
    assert fake_gh.secret_set_calls() == []


# --- _bot_handle_for unit tests ---------------------------------------------


def test_bot_handle_for_returns_slug_when_slug_is_set() -> None:
    """When a slug is available it takes priority over the scope owner."""
    scope = bot_setup.Scope(repo="owner/repo")
    assert bot_setup._bot_handle_for("acme-bot", scope) == "acme-bot"


def test_bot_handle_for_falls_back_to_repo_owner_when_slug_is_none() -> None:
    """With no slug, the owner extracted from the repo slug is used."""
    scope = bot_setup.Scope(repo="owner/repo")
    assert bot_setup._bot_handle_for(None, scope) == "owner"


def test_bot_handle_for_falls_back_to_org_when_slug_is_none() -> None:
    """With no slug and an org scope, the org login is used."""
    scope = bot_setup.Scope(org="acme-org")
    assert bot_setup._bot_handle_for(None, scope) == "acme-org"


def test_bot_handle_for_returns_slug_over_org_when_slug_is_set() -> None:
    """Slug wins even when the scope is org-scoped."""
    scope = bot_setup.Scope(org="acme-org")
    assert bot_setup._bot_handle_for("my-app-bot", scope) == "my-app-bot"
