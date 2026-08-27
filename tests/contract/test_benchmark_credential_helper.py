import http.server
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops


def _serve_bare_repo(root: Path) -> tuple[str, str]:
    """Create a bare repo with a main ref + a 401-challenging local HTTP server.

    Returns (bare_repo_path, ls-remote http url). The server 401s the first
    ``GET /repo.git/info/refs`` (query stripped) and serves the ref advertisement
    once an Authorization header is present — exactly the spike harness.
    """
    bare = root / "repo.git"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    subprocess.run(["git", "--git-dir", str(bare), "hash-object", "-w", "--stdin"],
                   check=True, capture_output=True, input=b"x\n")
    tree = subprocess.run(["git", "--git-dir", str(bare), "mktree"],
                          check=True, capture_output=True).stdout.decode().strip()
    commit = subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
         "--git-dir", str(bare), "commit-tree", tree, "-m", "c"],
        check=True, capture_output=True).stdout.decode().strip()
    subprocess.run(["git", "--git-dir", str(bare), "update-ref", "refs/heads/main", commit],
                   check=True, capture_output=True)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.split("?")[0].rstrip("/") != "/repo.git/info/refs":
                self.send_response(404)
                self.end_headers()
                return
            if self.headers.get("Authorization") is None:
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Basic realm="git"')
                self.end_headers()
                return
            body = subprocess.run(
                ["git", "--git-dir", str(bare), "for-each-ref", "--format=%(objectname)\t%(refname)"],
                check=True, capture_output=True).stdout
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a: Any) -> None:
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return str(bare), f"http://127.0.0.1:{srv.server_address[1]}/repo.git"


def test_git_ls_remote_drives_credential_helper_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    log = tmp_path / "helper.log"
    wrapper = tmp_path / "gh-cred-wrapper.sh"
    wrapper.write_text(
        '#!/bin/bash\n'
        f'LOG="{log}"\n'
        'echo "ARGV:$*" >> "$LOG"\n'
        'cat >> "$LOG"\n'
        'echo "---" >> "$LOG"\n'
        'if [ "${1:-}" = "get" ]; then\n'
        '  printf "protocol=http\\nhost=localhost\\nusername=u\\npassword=p\\n"\n'
        'fi\n'
    )
    wrapper.chmod(0o755)
    monkeypatch.setattr(git_ops, "GH_CREDENTIAL_HELPER", f"!{wrapper}")

    _, url = _serve_bare_repo(tmp_path)
    out = git_ops.git_ls_remote(tmp_path, url)

    assert "refs/heads/main" in out
    text = log.read_text()
    assert "ARGV:get" in text                     # Git invoked the helper with operation `get`
    assert "protocol=http" in text and "host=" in text   # protocol/host passed on stdin
    assert not (home / ".gitconfig").exists()     # command-scoped: no global config written
