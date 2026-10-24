"""Isolated regular-file reader; killed/reaped by the invocation on cancellation."""

from __future__ import annotations

import os
import stat
import sys


def main() -> None:
    # The parent grants execution only after spawn returns and authority is
    # checked again. A late/unreturned child can never begin filesystem access.
    if sys.stdin.buffer.read(1) != b"R":
        raise SystemExit(1)
    path, limit_text = sys.argv[1:]
    limit = int(limit_text)
    descriptors: list[int] = []
    try:
        directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        descriptors.append(directory)
        parts = path.split("/")[1:]
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("invalid path")
        for part in parts[:-1]:
            directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            descriptors.append(directory)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        descriptors.append(fd)
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
        sys.stdout.buffer.write(data)
    except (OSError, ValueError):
        raise SystemExit(1) from None
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


if __name__ == "__main__":
    main()
