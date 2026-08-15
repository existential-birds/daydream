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

# The uv bootstrap comes from a digest-pinned image stage, not pip: pip is the
# last moving installer a base build could still resolve at build time, and the
# digest names one specific multi-platform index for uv 0.11.29 that cannot
# drift. /uv and /uvx are copied onto PATH in the python stage below.
FROM ghcr.io/astral-sh/uv:0.11.29@sha256:eb2843a1e56fd9e30c7276ce1a52cba86e64c7b385f5e3279a0e08e02dd058fc AS uv

# The base is pinned by its immutable multi-platform index digest, not a mutable
# tag: a tag like `3.12-slim` tracks whatever `latest`-flavored manifest it
# currently resolves to, so a base rebuilt later could silently pick up a moved
# tag. The digest names one specific index for Python 3.12.13 slim and cannot
# drift.
FROM python:3.12.13-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36

# git: the repo image clones from an in-container mirror and daydream shells out
#   to git for every diff, patch and commit it takes.
# curl + ca-certificates: how the three CLI layers below fetch their releases.
# gnupg: verifies the signed Claude release manifest before its binary is trusted.
# procps: daydream's subprocess backends terminate and reap child CLIs; without
#   ps/kill a stalled-stream teardown is undiagnosable from inside the container.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      gnupg \
      procps \
 && rm -rf /var/lib/apt/lists/*

# uv installs the daydream wheel into the system interpreter. There is no venv
# on purpose: `daydream` must be on PATH for any process the harness starts,
# and a rollout container hosts exactly one application. The binary came from
# the digest-pinned uv image stage above, so the bootstrap tool cannot drift
# between builds.
COPY --from=uv /uv /uvx /bin/
RUN uv --version

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
# One static binary from Anthropic's signed release channel. Instead of piping
# an upstream installer script into bash, the manifest is verified against a
# pinned release-key fingerprint, the image's own Python extracts the expected
# checksum from the signed manifest, and the binary is checksum-verified before
# it is installed to /rollout/.local/bin/claude (already on PATH above). Pinned,
# because an unpinned agent CLI would silently change the rollout's tool loop
# between runs. Every failure here terminates the layer: no fallback release.
ARG INSTALL_CLAUDE=1
ARG CLAUDE_CODE_VERSION=2.1.214
RUN if [ "${INSTALL_CLAUDE}" = "1" ]; then \
      set -eu; \
      case "$(dpkg --print-architecture)" in \
        amd64) platform=linux-x64 ;; \
        arm64) platform=linux-arm64 ;; \
        *) echo "claude: unsupported architecture $(dpkg --print-architecture)" >&2; exit 1 ;; \
      esac; \
      release_fingerprint=31DDDE24DDFAB679F42D7BD2BAA929FF1A7ECACE; \
      tmp="$(mktemp -d)"; \
      export GNUPGHOME="${tmp}/gnupg"; \
      mkdir -m 0700 "${GNUPGHOME}"; \
      release_key="${tmp}/release_key"; \
      key_info="${tmp}/key_info"; \
      manifest="${tmp}/manifest"; \
      signature="${tmp}/signature"; \
      claude_path="${tmp}/claude"; \
      curl -fsSL https://downloads.claude.ai/keys/claude-code.asc -o "${release_key}"; \
      gpg --batch --with-colons --import-options show-only --import "${release_key}" 2>/dev/null \
        | awk -F: '$1 == "fpr" { print $10; exit }' > "${key_info}"; \
      actual_fingerprint="$(cat "${key_info}")"; \
      if [ "${actual_fingerprint}" != "${release_fingerprint}" ]; then \
        echo "claude: release key fingerprint mismatch: got ${actual_fingerprint}, want ${release_fingerprint}" >&2; \
        exit 1; \
      fi; \
      gpg --batch --import "${release_key}"; \
      release_url="https://downloads.claude.ai/claude-code-releases/${CLAUDE_CODE_VERSION}"; \
      curl -fsSL "${release_url}/manifest.json" -o "${manifest}"; \
      curl -fsSL "${release_url}/manifest.json.sig" -o "${signature}"; \
      gpg --batch --verify "${signature}" "${manifest}"; \
      checksum="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["platforms"][sys.argv[2]]["checksum"])' "${manifest}" "${platform}")"; \
      case "${checksum}" in \
        *[!0-9a-f]*) echo "claude: manifest checksum contains a non-hex character" >&2; exit 1 ;; \
      esac; \
      if [ "${#checksum}" -ne 64 ]; then \
        echo "claude: manifest checksum is not 64 hex chars: ${checksum}" >&2; exit 1; \
      fi; \
      curl -fsSL "${release_url}/${platform}/claude" -o "${claude_path}"; \
      printf '%s  %s\n' "${checksum}" "${claude_path}" | sha256sum -c -; \
      install -D -m 0755 "${claude_path}" /rollout/.local/bin/claude; \
      rm -rf "${tmp}"; \
      claude --version; \
    fi

# --- codex ----------------------------------------------------------------
# The standalone GitHub release binary rather than the npm package, so this layer
# does not drag a node runtime in ahead of the pi layer that actually needs one.
# Release tags are `rust-vX.Y.Z`; each asset unpacks to a single file named after
# its target triple, which is why it is renamed rather than extracted in place.
# The archive is downloaded to a temp file and verified against a locally pinned
# SHA256 constant (captured from the published GitHub release asset at pin time),
# then extracted — never piped straight into tar.
ARG INSTALL_CODEX=1
ARG CODEX_VERSION=0.145.0
RUN if [ "${INSTALL_CODEX}" = "1" ]; then \
      set -eu; \
      case "$(dpkg --print-architecture)" in \
        amd64) target=x86_64-unknown-linux-musl; checksum=bfaf13c9ba34f2ad764e4a916c49cf7177aeba329cf0f719e2227566fc8d662a ;; \
        arm64) target=aarch64-unknown-linux-musl; checksum=d384f90bc842450b42bd675feef06a12a46a3b1ca97efcb22566b270e4a11227 ;; \
        *) echo "codex: unsupported architecture $(dpkg --print-architecture)" >&2; exit 1 ;; \
      esac; \
      archive="/tmp/codex-${target}.tar.gz"; \
      curl -fsSL "https://github.com/openai/codex/releases/download/rust-v${CODEX_VERSION}/codex-${target}.tar.gz" -o "${archive}"; \
      printf '%s *%s\n' "${checksum}" "${archive}" | sha256sum -c -; \
      tar -xzf "${archive}" -C /usr/local/bin; \
      rm "${archive}"; \
      mv "/usr/local/bin/codex-${target}" /usr/local/bin/codex; \
      chmod +x /usr/local/bin/codex; \
      codex --version; \
    fi

# --- pi (and the node runtime it needs) -----------------------------------
# Heaviest, therefore last: pi ships only as an npm package, so this layer pulls
# a full node distribution. Node goes to /usr/local so `npm install -g` puts the
# `pi` binary in /usr/local/bin with no prefix juggling. The .tar.gz build is
# used, not .tar.xz, so the base needs no xz-utils. The Node archive is
# downloaded to a temp file and verified against a locally pinned SHA256 constant
# (captured from v22.17.1's published signed SHASUMS256.txt at pin time) before it
# is extracted. The pi package tarball is likewise downloaded to a temp file and
# verified against a locally pinned SHA256 constant (captured from the npm
# registry tarball at pin time) before npm installs it — no streaming install
# from the registry whose bytes were never pinned.
ARG INSTALL_PI=1
ARG NODE_VERSION=22.17.1
ARG PI_VERSION=0.82.1
RUN if [ "${INSTALL_PI}" = "1" ]; then \
      set -eu; \
      case "$(dpkg --print-architecture)" in \
        amd64) node_arch=x64; checksum=cfb6ac0cf339825fe36efd1f18a79016b02aca19fbfa6c9547c57e27dc09f6ea ;; \
        arm64) node_arch=arm64; checksum=f53510706998cf044f634190416f0588e7e1937aecea938768952e0f0ac1f41b ;; \
        *) echo "pi: unsupported architecture $(dpkg --print-architecture)" >&2; exit 1 ;; \
      esac; \
      archive="/tmp/node-v${NODE_VERSION}-linux-${node_arch}.tar.gz"; \
      curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${node_arch}.tar.gz" -o "${archive}"; \
      printf '%s *%s\n' "${checksum}" "${archive}" | sha256sum -c -; \
      tar -xzf "${archive}" -C /usr/local --strip-components=1 \
          --exclude CHANGELOG.md --exclude LICENSE --exclude README.md; \
      rm "${archive}"; \
      pi_checksum=8343ab95cbab5766f2f5d48844df8db13e772ead2e2976166cbb820a29dacb7d; \
      pi_tarball="/tmp/pi-coding-agent-${PI_VERSION}.tgz"; \
      curl -fsSL "https://registry.npmjs.org/@earendil-works/pi-coding-agent/-/pi-coding-agent-${PI_VERSION}.tgz" -o "${pi_tarball}"; \
      printf '%s *%s\n' "${pi_checksum}" "${pi_tarball}" | sha256sum -c -; \
      npm install -g --no-fund --no-audit "${pi_tarball}"; \
      rm "${pi_tarball}"; \
      npm cache clean --force; \
      pi --version; \
    fi

# The checkout in the repo image is created by root during the build and read by
# whatever uid the rollout runs as; git refuses a repository it does not own, and
# that refusal would surface as an unreviewable empty diff rather than an error.
RUN git config --global --add safe.directory '*'

WORKDIR /work
