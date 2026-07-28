# Rollout base image: daydream plus the CLIs its backends drive.
#
# Nothing repo-specific lives here. One base is shared by every PR-snapshot image
# (images/repo.Dockerfile), so a corpus of N pull requests pays for these layers
# once instead of N times.
#
# Every backend CLI is installed in its OWN layer behind its OWN build arg, in
# cheapest-first order (claude, codex, pi+node). Two reasons: re-pinning one CLI
# must not invalidate the others' layers, and an operator who only ever runs
# `--backend claude` can pass INSTALL_CODEX=0 INSTALL_PI=0 and stop paying for a
# node runtime it will never execute. The harness enforces the other half of that
# contract at rollout time — DaydreamReviewHarness.setup() refuses an image whose
# selected backend's binary is missing (daydream_review_v1/harness.py:66-79).

FROM python:3.12-slim

# git: the repo image clones from an in-container mirror and daydream shells out
#   to git for every diff, patch and commit it takes.
# curl + ca-certificates: how the three CLI layers below fetch their releases.
# procps: daydream's subprocess backends terminate and reap child CLIs; without
#   ps/kill a stalled-stream teardown is undiagnosable from inside the container.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      procps \
 && rm -rf /var/lib/apt/lists/*

# uv installs the daydream wheel into the system interpreter. There is no venv
# on purpose: `daydream` must be on PATH for any process the harness starts,
# and a rollout container hosts exactly one application.
RUN pip install --no-cache-dir uv

# HOME is set before the CLI installs, not after, so the claude installer writes
# its binary and its cache under the SAME home the harness later hands daydream
# (daydream_review_v1/backends.py ROLLOUT_HOME = /rollout). Installing under a
# different HOME would put `claude` somewhere the rollout cannot see it.
ENV HOME=/rollout
ENV PATH=/rollout/.local/bin:${PATH}

# /rollout/archive is DAYDREAM_ARCHIVE_DIR (rundir.DEFAULT_ARCHIVE_ROOT) — the
# reward reads the archived run dir back out of it. /work holds the checkout the
# repo image clones into (taskset.DEFAULT_REPO_PATH = /work/repo).
RUN mkdir -p /rollout/archive /work

ARG DAYDREAM_WHEEL
COPY dist/${DAYDREAM_WHEEL} /tmp/
# `command -v`, not `daydream --version`: the CLI has no --version flag, and the
# only thing this layer needs to prove is that the console script is on PATH,
# which is exactly what DaydreamReviewHarness.setup() checks at rollout time.
RUN uv pip install --system /tmp/${DAYDREAM_WHEEL} \
 && rm /tmp/${DAYDREAM_WHEEL} \
 && command -v daydream

# --- claude ---------------------------------------------------------------
# Cheapest of the three: one static binary from Anthropic's own installer, which
# drops it at $HOME/.local/bin/claude (already on PATH above). Pinned, because an
# unpinned agent CLI would silently change the rollout's tool loop between runs.
ARG INSTALL_CLAUDE=1
ARG CLAUDE_CODE_VERSION=2.1.214
RUN if [ "${INSTALL_CLAUDE}" = "1" ]; then \
      set -eu; \
      curl -fsSL https://claude.ai/install.sh | bash -s "${CLAUDE_CODE_VERSION}"; \
      claude --version; \
    fi

# --- codex ----------------------------------------------------------------
# The standalone GitHub release binary rather than the npm package, so this layer
# does not drag a node runtime in ahead of the pi layer that actually needs one.
# Release tags are `rust-vX.Y.Z`; each asset unpacks to a single file named after
# its target triple, which is why it is renamed rather than extracted in place.
ARG INSTALL_CODEX=1
ARG CODEX_VERSION=0.145.0
RUN if [ "${INSTALL_CODEX}" = "1" ]; then \
      set -eu; \
      case "$(dpkg --print-architecture)" in \
        amd64) target=x86_64-unknown-linux-musl ;; \
        arm64) target=aarch64-unknown-linux-musl ;; \
        *) echo "codex: unsupported architecture $(dpkg --print-architecture)" >&2; exit 1 ;; \
      esac; \
      curl -fsSL "https://github.com/openai/codex/releases/download/rust-v${CODEX_VERSION}/codex-${target}.tar.gz" \
        | tar -xz -C /usr/local/bin; \
      mv "/usr/local/bin/codex-${target}" /usr/local/bin/codex; \
      chmod +x /usr/local/bin/codex; \
      codex --version; \
    fi

# --- pi (and the node runtime it needs) -----------------------------------
# Heaviest, therefore last: pi ships only as an npm package, so this layer pulls
# a full node distribution. Node goes to /usr/local so `npm install -g` puts the
# `pi` binary in /usr/local/bin with no prefix juggling. The .tar.gz build is
# used, not .tar.xz, so the base needs no xz-utils.
ARG INSTALL_PI=1
ARG NODE_VERSION=22.17.1
ARG PI_VERSION=0.82.1
RUN if [ "${INSTALL_PI}" = "1" ]; then \
      set -eu; \
      case "$(dpkg --print-architecture)" in \
        amd64) node_arch=x64 ;; \
        arm64) node_arch=arm64 ;; \
        *) echo "pi: unsupported architecture $(dpkg --print-architecture)" >&2; exit 1 ;; \
      esac; \
      curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${node_arch}.tar.gz" \
        | tar -xz -C /usr/local --strip-components=1 \
            --exclude CHANGELOG.md --exclude LICENSE --exclude README.md; \
      npm install -g --no-fund --no-audit "@earendil-works/pi-coding-agent@${PI_VERSION}"; \
      npm cache clean --force; \
      pi --version; \
    fi

# The checkout in the repo image is created by root during the build and read by
# whatever uid the rollout runs as; git refuses a repository it does not own, and
# that refusal would surface as an unreviewable empty diff rather than an error.
RUN git config --global --add safe.directory '*'

WORKDIR /work
