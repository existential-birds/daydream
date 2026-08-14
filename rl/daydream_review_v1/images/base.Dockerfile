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
# Cheapest of the three: one static binary from Anthropic's raw release, dropped
# at $HOME/.local/bin/claude (already on PATH above). Pinned, because an unpinned
# agent CLI would silently change the rollout's tool loop between runs.
#
# The download is verified against hard-coded digests before it is installed.
# Each digest is the `platforms.linux-x64.checksum` / `platforms.linux-arm64.checksum`
# from https://downloads.claude.ai/claude-code-releases/${CLAUDE_CODE_VERSION}/manifest.json.
# Bump CLAUDE_CODE_VERSION and BOTH digests together, from that same manifest.
ARG INSTALL_CLAUDE=1
ARG CLAUDE_CODE_VERSION=2.1.214
ARG CLAUDE_CODE_SHA256_AMD64=3c029136f7c81f54ed4a38e9d52e655aad536433dbbde50519c8c31bb646ad14
ARG CLAUDE_CODE_SHA256_ARM64=4c38f26a57a42619ee813f15dc39fc1fa4fe0bb403215c3cdc342b58fa689c3c
RUN if [ "${INSTALL_CLAUDE}" = "1" ]; then \
      set -eu; \
      case "$(dpkg --print-architecture)" in \
        amd64) platform=linux-x64; checksum="${CLAUDE_CODE_SHA256_AMD64}" ;; \
        arm64) platform=linux-arm64; checksum="${CLAUDE_CODE_SHA256_ARM64}" ;; \
        *) echo "claude: unsupported architecture $(dpkg --print-architecture)" >&2; exit 1 ;; \
      esac; \
      mkdir -p /rollout/.local/bin; \
      curl -fsSL -o /tmp/claude "https://downloads.claude.ai/claude-code-releases/${CLAUDE_CODE_VERSION}/${platform}/claude"; \
      printf '%s  %s\n' "${checksum}" /tmp/claude | sha256sum --check --strict -; \
      install -m 0755 /tmp/claude /rollout/.local/bin/claude; \
      rm -f /tmp/claude; \
      claude --version; \
    fi

# --- codex ----------------------------------------------------------------
# The standalone GitHub release binary rather than the npm package, so this layer
# does not drag a node runtime in ahead of the pi layer that actually needs one.
# Release tags are `rust-vX.Y.Z`; each asset unpacks to a single file named after
# its target triple, which is why it is renamed rather than extracted in place.
#
# The download is verified against hard-coded digests before it is extracted. Each
# digest is the 64-hex `digest` of the `codex-x86_64-unknown-linux-musl.tar.gz` /
# `codex-aarch64-unknown-linux-musl.tar.gz` asset from
# https://api.github.com/repos/openai/codex/releases/tags/rust-v${CODEX_VERSION}.
# Bump CODEX_VERSION and BOTH digests together, from that same release metadata.
ARG INSTALL_CODEX=1
ARG CODEX_VERSION=0.145.0
ARG CODEX_SHA256_AMD64=bfaf13c9ba34f2ad764e4a916c49cf7177aeba329cf0f719e2227566fc8d662a
ARG CODEX_SHA256_ARM64=d384f90bc842450b42bd675feef06a12a46a3b1ca97efcb22566b270e4a11227
RUN if [ "${INSTALL_CODEX}" = "1" ]; then \
      set -eu; \
      case "$(dpkg --print-architecture)" in \
        amd64) target=x86_64-unknown-linux-musl; checksum="${CODEX_SHA256_AMD64}" ;; \
        arm64) target=aarch64-unknown-linux-musl; checksum="${CODEX_SHA256_ARM64}" ;; \
        *) echo "codex: unsupported architecture $(dpkg --print-architecture)" >&2; exit 1 ;; \
      esac; \
      curl -fsSL -o /tmp/codex.tar.gz "https://github.com/openai/codex/releases/download/rust-v${CODEX_VERSION}/codex-${target}.tar.gz"; \
      printf '%s  %s\n' "${checksum}" /tmp/codex.tar.gz | sha256sum --check --strict -; \
      tar -xzf /tmp/codex.tar.gz -C /usr/local/bin; \
      rm -f /tmp/codex.tar.gz; \
      mv "/usr/local/bin/codex-${target}" /usr/local/bin/codex; \
      chmod +x /usr/local/bin/codex; \
      codex --version; \
    fi

# --- pi (and the node runtime it needs) -----------------------------------
# Heaviest, therefore last: pi ships only as an npm package, so this layer pulls
# a full node distribution. Node goes to /usr/local so `npm install -g` puts the
# `pi` binary in /usr/local/bin with no prefix juggling. The .tar.gz build is
# used, not .tar.xz, so the base needs no xz-utils.
#
# The download is verified against hard-coded digests before it is extracted. Each
# digest is the `node-v${NODE_VERSION}-linux-x64.tar.gz` / `...-linux-arm64.tar.gz`
# row from https://nodejs.org/dist/v${NODE_VERSION}/SHASUMS256.txt.
# Bump NODE_VERSION and BOTH digests together, from that same SHASUMS256.txt.
ARG INSTALL_PI=1
ARG NODE_VERSION=22.17.1
ARG NODE_SHA256_AMD64=cfb6ac0cf339825fe36efd1f18a79016b02aca19fbfa6c9547c57e27dc09f6ea
ARG NODE_SHA256_ARM64=f53510706998cf044f634190416f0588e7e1937aecea938768952e0f0ac1f41b
ARG PI_VERSION=0.82.1
RUN if [ "${INSTALL_PI}" = "1" ]; then \
      set -eu; \
      case "$(dpkg --print-architecture)" in \
        amd64) node_arch=x64; checksum="${NODE_SHA256_AMD64}" ;; \
        arm64) node_arch=arm64; checksum="${NODE_SHA256_ARM64}" ;; \
        *) echo "pi: unsupported architecture $(dpkg --print-architecture)" >&2; exit 1 ;; \
      esac; \
      curl -fsSL -o /tmp/node.tar.gz "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${node_arch}.tar.gz"; \
      printf '%s  %s\n' "${checksum}" /tmp/node.tar.gz | sha256sum --check --strict -; \
      tar -xzf /tmp/node.tar.gz -C /usr/local --strip-components=1 \
            --exclude CHANGELOG.md --exclude LICENSE --exclude README.md; \
      rm -f /tmp/node.tar.gz; \
      npm install -g --no-fund --no-audit "@earendil-works/pi-coding-agent@${PI_VERSION}"; \
      npm cache clean --force; \
      pi --version; \
    fi

# The checkout in the repo image is created by root during the build and read by
# whatever uid the rollout runs as; git refuses a repository it does not own, and
# that refusal would surface as an unreviewable empty diff rather than an error.
RUN git config --global --add safe.directory '*'

WORKDIR /work
