"""Hold real isolated readers at filesystem access or unfinished startup."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest


@asynccontextmanager
async def gated_reader(monkeypatch: pytest.MonkeyPatch, root: Path, source: Path, *, late_spawn: bool = False,
) -> AsyncIterator[tuple[asyncio.Event, asyncio.Event, list[asyncio.subprocess.Process], Path]]:
    """Inject a FIFO wait at the actual final os.open; never substitute read results."""
    entered, release = asyncio.Event(), asyncio.Event()
    marker, fifo = root / "reader-entered", root / "reader-release"
    os.mkfifo(fifo)
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    loop = asyncio.get_running_loop()
    def notify() -> None:
        loop.remove_reader(read_fd)
        entered.set()
    loop.add_reader(read_fd, notify)
    children: list[asyncio.subprocess.Process] = []
    spawn = asyncio.create_subprocess_exec
    script = ("import os,sys,runpy; original=os.open\n"
        "def gated(path,*args,**kwargs):\n"
        f" if path == {source.name!r}:\n"
        f"  open({str(marker)!r},'w').write(str(os.getpid()))\n"
        f"  os.write({write_fd},b'R'); os.read(original({str(fifo)!r},os.O_RDWR),1)\n"
        " return original(path,*args,**kwargs)\n"
        "os.open=gated; script=sys.argv.pop(1); runpy.run_path(script,run_name='__main__')\n")

    async def launch(*args: Any, **options: Any) -> asyncio.subprocess.Process:
        reader = len(args) > 2 and args[1] == "-I" and Path(args[2]).name == "host_tools_read.py"
        child = await spawn(*([sys.executable, "-I", "-c", script, *args[2:]] if reader else args),
                            **({**options, "pass_fds": (write_fd,)} if reader else options))
        if reader:
            children.append(child)
            if late_spawn:
                entered.set()
                await release.wait()
        return child

    monkeypatch.setattr("daydream.host_tools.asyncio.create_subprocess_exec", launch)
    try:
        yield entered, release, children, marker
    finally:
        release.set()
        for child in children:
            if child.returncode is None:
                child.kill()
        await asyncio.gather(*(child.wait() for child in children))
        loop.remove_reader(read_fd)
        os.close(read_fd)
        os.close(write_fd)
