"""Real-path tests for :mod:`daydream.bot_setup`.

Task 5 covers the App-from-manifest localhost registration seam. The live
browser/manifest leg (binding a port, opening a browser, blocking on the
callback) is isolated behind :class:`daydream.bot_setup._ManifestListener` so
the code-exchange behavior is testable without real GitHub: the test drives
``_handle_code`` directly, monkeypatching only the manifest-code exchange.
"""

from pathlib import Path

import pytest

from daydream import bot_setup, config, git_ops
from daydream.github_app import AppCredentials, GitHubAppError
from tests.harness.fake_gh import FakeGh, install_fake_gh


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
