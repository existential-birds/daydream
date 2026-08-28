FROM __BASE_IMAGE__

# Separate verifier environment: stdlib plus the hash-locked HTTP client, a
# version-pinned Node runtime and the version-pinned Claude Code CLI (the
# claude-cli judge path). httpx and its transitive subtree are pinned
# version+content-hash with the committed closure (same as
# runtime-requirements.lock) and installed with --require-hashes, and the CLI
# installs via `npm ci` from a committed package-lock.json so every transitive
# is version- and integrity-pinned rather than re-resolved from the registry
# at build time; the verifier -- the container carrying the judge API key to
# the configured allowlisted endpoint -- builds deterministically. Node and
# the CLI add a few hundred MB to the image; accepted for this purpose.
RUN printf 'httpx==0.28.1 --hash=sha256:75e98c5f16b0f35b567856f597f06ff2270a374470a5c2392242528e3e3e42fc --hash=sha256:d909fcccc110f8c7faf814ca82a9a4d816bc5a6dbfea25d6591d6985b8ba59ad\nhttpcore==1.0.9 --hash=sha256:2d400746a40668fc9dec9810239072b40b4484b640a8c38fd654a024c7a1bf55 --hash=sha256:6e34463af53fd2ab5d807f399a9b45ea31c3dfa2276f15a2c3f00afff6e176e8\nanyio==4.12.1 --hash=sha256:41cfcc3a4c85d3f05c932da7c26d0201ac36f72abd4435ba90d0464a3ffed703 --hash=sha256:d405828884fc140aa80a3c667b8beed277f1dfedec42ba031bd6ac3db606ab6c\ncertifi==2026.1.4 --hash=sha256:9943707519e4add1115f44c2bc244f782c0249876bf51b6599fee1ffbedd685c --hash=sha256:ac726dd470482006e014ad384921ed6438c457018f4b3d204aea4281258b2120\nh11==0.16.0 --hash=sha256:4e35b956cf45792e4caa5885e69fba00bdbc6ffafbfa020300e549b208ee5ff1 --hash=sha256:63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86\nidna==3.15 --hash=sha256:048adeaf8c2d788c40fee287673ccaa74c24ffd8dcf09ffa555a2fbb59f10ac8 --hash=sha256:ca962446ea538f7092a95e057da437618e886f4d349216d2b1e294abfdb65fdc\ntyping_extensions==4.15.0 --hash=sha256:0cea48d173cc12fa28ecabc3b837ea3cf6f38c6d1136f85cbaaf598984861466 --hash=sha256:f0fa19c6845758ab08074a0cfa8b7aecb71c999ca73d62883bc25cc018c4e548\n' > /tmp/httpx-lock.txt \
    && python -m pip install --no-cache-dir --require-hashes -r /tmp/httpx-lock.txt \
    && rm /tmp/httpx-lock.txt

# Node v22.14.0 LTS + @anthropic-ai/claude-code@2.1.250, both version-pinned;
# the tarball is checksum-verified against its published SHASUMS256 value and
# the CLI installs with `npm ci` from the embedded package-lock.json (every
# transitive version- and integrity-pinned, like the httpx closure), so a
# re-published tarball fails the build instead of silently changing what ships.
RUN set -eux; \
    case "$(uname -m)" in \
        x86_64) narch=x64; nsha=69b09dba5c8dcb05c4e4273a4340db1005abeafe3927efda2bc5b249e80437ec;; \
        aarch64) narch=arm64; nsha=08bfbf538bad0e8cbb0269f0173cca28d705874a67a22f60b57d99dc99e30050;; \
        *) echo "unsupported arch"; exit 1;; \
    esac; \
    # python:3.12-slim ships no curl/xz-utils; use the Python stdlib for both
    # the download and the xz-compressed tar extraction. The socket default
    # timeout bounds every network operation, so a stalled nodejs.org
    # connection fails the build instead of hanging it indefinitely.
    python -c "import socket,urllib.request,sys; socket.setdefaulttimeout(60); urllib.request.urlretrieve('https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-'+sys.argv[1]+'.tar.xz', '/tmp/node.tar.xz')" "$narch"; \
    echo "${nsha}  /tmp/node.tar.xz" | sha256sum -c -; \
    mkdir -p /usr/local/lib/nodejs; \
    python -c "import tarfile; tarfile.open('/tmp/node.tar.xz').extractall('/usr/local/lib/nodejs', filter='data')"; \
    rm -f /tmp/node.tar.xz; \
    ln -s "/usr/local/lib/nodejs/node-v22.14.0-linux-${narch}/bin/node" /usr/local/bin/node; \
    ln -s "/usr/local/lib/nodejs/node-v22.14.0-linux-${narch}/bin/npm" /usr/local/bin/npm; \
    ln -s "/usr/local/lib/nodejs/node-v22.14.0-linux-${narch}/bin/npx" /usr/local/bin/npx; \
    mkdir -p /tmp/claude-cli; \
    cat > /tmp/claude-cli/package.json <<'EOF'
{
  "name": "claude-cli-verifier",
  "private": true,
  "version": "1.0.0",
  "dependencies": {
    "@anthropic-ai/claude-code": "2.1.250"
  }
}
EOF
    cat > /tmp/claude-cli/package-lock.json <<'EOF'
{
  "name": "claude-cli-verifier",
  "version": "1.0.0",
  "lockfileVersion": 3,
  "requires": true,
  "packages": {
    "": {
      "name": "claude-cli-verifier",
      "version": "1.0.0",
      "dependencies": {
        "@anthropic-ai/claude-code": "2.1.250"
      }
    },
    "node_modules/@anthropic-ai/claude-code": {
      "version": "2.1.250",
      "resolved": "https://registry.npmjs.org/@anthropic-ai/claude-code/-/claude-code-2.1.250.tgz",
      "integrity": "sha512-SsqdHBePKgpoquZdMPbwSZZc2PkQy/qBcekroihILhRdPAAtpXXH49POHl7e6bujj55MaKPk4Wa7gpFXuMKwnA==",
      "hasInstallScript": true,
      "license": "SEE LICENSE IN README.md",
      "bin": {
        "claude": "bin/claude.exe"
      },
      "engines": {
        "node": ">=22.0.0"
      },
      "optionalDependencies": {
        "@anthropic-ai/claude-code-darwin-arm64": "2.1.250",
        "@anthropic-ai/claude-code-darwin-x64": "2.1.250",
        "@anthropic-ai/claude-code-linux-arm64": "2.1.250",
        "@anthropic-ai/claude-code-linux-arm64-musl": "2.1.250",
        "@anthropic-ai/claude-code-linux-x64": "2.1.250",
        "@anthropic-ai/claude-code-linux-x64-musl": "2.1.250",
        "@anthropic-ai/claude-code-win32-arm64": "2.1.250",
        "@anthropic-ai/claude-code-win32-x64": "2.1.250"
      }
    },
    "node_modules/@anthropic-ai/claude-code-darwin-arm64": {
      "version": "2.1.250",
      "resolved": "https://registry.npmjs.org/@anthropic-ai/claude-code-darwin-arm64/-/claude-code-darwin-arm64-2.1.250.tgz",
      "integrity": "sha512-THGQj2+aPMXrHpEbBFE/4PzJYK1Y9NlKdo+KhizGK020pZMBLP11KyeeBTm8HUJ5ZHeWIyB7ohwbQoZ7ttKGPA==",
      "cpu": [
        "arm64"
      ],
      "license": "SEE LICENSE IN LICENSE.md",
      "optional": true,
      "os": [
        "darwin"
      ]
    },
    "node_modules/@anthropic-ai/claude-code-darwin-x64": {
      "version": "2.1.250",
      "resolved": "https://registry.npmjs.org/@anthropic-ai/claude-code-darwin-x64/-/claude-code-darwin-x64-2.1.250.tgz",
      "integrity": "sha512-IF1MlGV5zT60a2fCsRgLyuvlRODrfMfKfhfvdqxPgax0bNaatqzqodZ9n23o2m2QIksHo5HF3T2Oe3frE+8l6Q==",
      "cpu": [
        "x64"
      ],
      "license": "SEE LICENSE IN LICENSE.md",
      "optional": true,
      "os": [
        "darwin"
      ]
    },
    "node_modules/@anthropic-ai/claude-code-linux-arm64": {
      "version": "2.1.250",
      "resolved": "https://registry.npmjs.org/@anthropic-ai/claude-code-linux-arm64/-/claude-code-linux-arm64-2.1.250.tgz",
      "integrity": "sha512-NVbdmP9kLenUGdPjSS9p+XTY0CqC+pjWo3sJMM4DrCOlUeJLqKcAB9f9+IvG5PHBAfDbcEstXjeGqDVavsvLMA==",
      "cpu": [
        "arm64"
      ],
      "license": "SEE LICENSE IN LICENSE.md",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@anthropic-ai/claude-code-linux-arm64-musl": {
      "version": "2.1.250",
      "resolved": "https://registry.npmjs.org/@anthropic-ai/claude-code-linux-arm64-musl/-/claude-code-linux-arm64-musl-2.1.250.tgz",
      "integrity": "sha512-GSDQ9LSnHojpXCcI/Zru8wWIoa+Kbgk4aHpc3/AneuGgGbNKbnj2Hi9awLYSuOqRg+MmmeQCM+r/4gCUSqyOrA==",
      "cpu": [
        "arm64"
      ],
      "license": "SEE LICENSE IN LICENSE.md",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@anthropic-ai/claude-code-linux-x64": {
      "version": "2.1.250",
      "resolved": "https://registry.npmjs.org/@anthropic-ai/claude-code-linux-x64/-/claude-code-linux-x64-2.1.250.tgz",
      "integrity": "sha512-8bNQMnhl9p1j/xQ36vj/ghZrPQo1sm5OUJKbG+kQSVfOzhpsKWnOtrTY/xzABqUDd6lY6r7Tom7O9IXOq6gsHA==",
      "cpu": [
        "x64"
      ],
      "license": "SEE LICENSE IN LICENSE.md",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@anthropic-ai/claude-code-linux-x64-musl": {
      "version": "2.1.250",
      "resolved": "https://registry.npmjs.org/@anthropic-ai/claude-code-linux-x64-musl/-/claude-code-linux-x64-musl-2.1.250.tgz",
      "integrity": "sha512-1IQRUb51+KckMKTuNUJb+666ioPsY0E5NR5jI2iSKfdkzu5iRjyXQtOkn5qSUnRh7DPobx64g7sZHWJQoUvOuQ==",
      "cpu": [
        "x64"
      ],
      "license": "SEE LICENSE IN LICENSE.md",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@anthropic-ai/claude-code-win32-arm64": {
      "version": "2.1.250",
      "resolved": "https://registry.npmjs.org/@anthropic-ai/claude-code-win32-arm64/-/claude-code-win32-arm64-2.1.250.tgz",
      "integrity": "sha512-e1GS0M6TXnZK17xwSxmPGR3RtErMKp9vyYcEt3iMLk5aae/gl+dvIQxzu4uNawPBtfQehygI1tHbArIEWd9f7g==",
      "cpu": [
        "arm64"
      ],
      "license": "SEE LICENSE IN LICENSE.md",
      "optional": true,
      "os": [
        "win32"
      ]
    },
    "node_modules/@anthropic-ai/claude-code-win32-x64": {
      "version": "2.1.250",
      "resolved": "https://registry.npmjs.org/@anthropic-ai/claude-code-win32-x64/-/claude-code-win32-x64-2.1.250.tgz",
      "integrity": "sha512-LguXPu3lFuByqyQOy0xIB2XhFl6jlODUSP81xt9Xpgf5Ha7YyOLg7g1lSkZXfWx7jpNq6E5QtFbsfyJzSFsFvg==",
      "cpu": [
        "x64"
      ],
      "license": "SEE LICENSE IN LICENSE.md",
      "optional": true,
      "os": [
        "win32"
      ]
    }
  }
}
EOF
    (cd /tmp/claude-cli && npm ci --no-audit --no-fund); \
    ln -s /tmp/claude-cli/node_modules/.bin/claude /usr/local/bin/claude; \
    node --version && claude --version

WORKDIR /tests
COPY test.sh score_review.py verifier_core.py judge_prompt.md golden-review.json verifier-metadata.json ./
RUN chmod +x test.sh
