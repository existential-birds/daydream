"""Isolated persistent regular-file reader; each parent command is a read gate."""

from __future__ import annotations

import json
import os
import stat
import sys
from contextlib import ExitStack


def read_file(path: str, limit: int) -> bytes:
    with ExitStack() as descriptors:
        directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        descriptors.callback(os.close, directory)
        parts = path.split("/")[1:]
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("invalid path")
        for part in parts[:-1]:
            directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            descriptors.callback(os.close, directory)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        descriptors.callback(os.close, fd)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("nonregular file")
        data = bytearray()
        while len(data) <= limit:
            chunk = os.read(fd, min(4096, limit + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) > limit:
            raise ValueError("oversized file")
        return bytes(data)


def main() -> None:
    limit = int(sys.argv[1])
    while command := sys.stdin.buffer.readline(16385):
        if len(command) > 16384 or not command.endswith(b"\n"):
            raise SystemExit(1)
        try:
            path = json.loads(command)
            if not isinstance(path, str):
                raise ValueError("invalid command")
            data = read_file(path, limit)
            sys.stdout.buffer.write(str(len(data)).encode() + b"\n" + data)
        except (OSError, ValueError):
            sys.stdout.buffer.write(b"E\n")
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
