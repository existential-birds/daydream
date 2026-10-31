#!/usr/bin/env python3
"""Offline capture/admission benchmark. Run old/new code against identical private fixtures.

Set DAYDREAM_CAPTURE_BENCH_CODE to the import checkout (defaults to this checkout) and
DAYDREAM_CAPTURE_BENCH_ROOT to the external fixture/result directory.
Python allocations and timings run separately; macOS process footprint is sampled
with the same RUSAGE_INFO_V0 method as the original investigation.
No providers run; only disposable fixture refs/index are written.
"""

import argparse
import contextlib
import functools
import gc
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHECKOUT = Path(os.environ.get("DAYDREAM_CAPTURE_BENCH_CODE", Path(__file__).resolve().parents[1])).resolve()
ROOT = Path(os.environ["DAYDREAM_CAPTURE_BENCH_ROOT"]).resolve()
if ROOT == CHECKOUT or ROOT.is_relative_to(CHECKOUT):
    raise SystemExit("benchmark files must stay outside the implementation checkout")
ROOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CHECKOUT))
from daydream.git_ops import captured_source as cs, immutable, process  # noqa: E402 - selected benchmark implementation


def git_environment():
    """Keep fixture Git operations independent of inherited Git routing and configuration."""
    return {
        **{key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    }


def run_git(repo, *args, data=None):
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        input=data,
        capture_output=True,
        check=True,
        env=git_environment(),
    ).stdout


def obj(repo, kind, body):
    raw = kind.encode() + b" " + str(len(body)).encode() + b"\0" + body
    oid = hashlib.sha1(raw).hexdigest()
    path = repo / ".git/objects" / oid[:2] / oid[2:]
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(zlib.compress(raw))
    return oid


def tree(repo, entries):
    return obj(
        repo,
        "tree",
        b"".join(b"100644 " + name.encode() + b"\0" + bytes.fromhex(oid) for name, oid in sorted(entries.items())),
    )


def fixture(
    name,
    source_bytes=1024 * 1024,
    files=16,
    changes=1,
    patch_lines=1,
    history_mib=0,
    layout="loose",
    oversized_store=False,
    oversized_source=False,
):
    repo = ROOT / "fixtures" / name
    if repo.exists():
        return json.loads((repo.parent / (name + ".json")).read_text())
    repo.mkdir(parents=True)
    run_git(repo, "init", "--template=", ".")
    rng = random.Random(2176)
    before = {}
    after = {}
    per_file = max(32, (source_bytes - changes * 128) // max(1, files - changes))
    for i in range(files):
        path = f"f{i:05d}.txt"
        # UTF-8 LF text, unique bodies, deliberately not incompressible source.
        prefix = f"file {i:05d} unchanged supporting context\n".encode()
        leaf_bytes = 128 if i < changes else per_file
        body = (prefix + b"x" * max(0, leaf_bytes - len(prefix) - 1) + b"\n")[:leaf_bytes]
        if i < changes:
            body = b"old line\n" * patch_lines + body
        before[path] = obj(repo, "blob", body)
        new = (b"new line\n" * patch_lines + body[9 * patch_lines :]) if i < changes else body
        after[path] = obj(repo, "blob", new)
        (repo / path).write_bytes(new)
    bt, ht = tree(repo, before), tree(repo, after)
    base = obj(
        repo,
        "commit",
        (
            f"tree {bt}\nauthor Offline <offline@invalid> 0 +0000\n"
            "committer Offline <offline@invalid> 0 +0000\n\nbase\n"
        ).encode(),
    )
    head = obj(
        repo,
        "commit",
        (
            f"tree {ht}\nparent {base}\nauthor Offline <offline@invalid> 0 +0000\n"
            "committer Offline <offline@invalid> 0 +0000\n\nhead\n"
        ).encode(),
    )
    (repo / ".git/HEAD").write_text(head + "\n")
    run_git(repo, "read-tree", head)
    # Reachable on a separate fixture ref, independent of compared commits/trees.
    history = head
    for i in range(history_mib):
        oid = obj(repo, "blob", rng.randbytes(1024 * 1024))
        ht2 = tree(repo, {"unrelated.bin": oid})
        history = obj(
            repo,
            "commit",
            (
                f"tree {ht2}\nparent {history}\nauthor Offline <offline@invalid> 0 +0000\n"
                f"committer Offline <offline@invalid> 0 +0000\n\nhistory {i}\n"
            ).encode(),
        )
    (repo / ".git/refs/heads").mkdir(parents=True, exist_ok=True)
    (repo / ".git/refs/heads/history").write_text(history + "\n")
    if layout == "packed":
        run_git(repo, "repack", "-a", "-d", "--window=0", "--threads=1")
    if oversized_store:
        p = repo / ".git/objects/ff" / ("f" * 38)
        p.parent.mkdir(exist_ok=True)
        with p.open("wb") as f:
            f.truncate(2 * 1024**3 + 1)
    if oversized_source:
        with (repo / "oversized.txt").open("wb") as f:
            f.truncate(256 * 1024**2 + 1)
    info = dict(
        name=name,
        repo=str(repo),
        base=base,
        head=head,
        source_bytes=source_bytes,
        files=files,
        changes=changes,
        patch_lines=patch_lines,
        history_mib=history_mib,
        layout=layout,
        oversized_store=oversized_store,
        oversized_source=oversized_source,
    )
    info["source_objects"] = disk_stats(repo / ".git/objects")
    (repo.parent / (name + ".json")).write_text(json.dumps(info, indent=2))
    return info


def disk_stats(root):
    logical = allocated = count = 0
    if root.exists():
        for parent, dirs, files in os.walk(root):
            for name in files:
                try:
                    s = (Path(parent) / name).lstat()
                except FileNotFoundError:
                    continue
                logical += s.st_size
                allocated += s.st_blocks * 512
                count += 1
    return dict(logical=logical, allocated=allocated, files=count)


class Metrics:
    def __init__(self):
        self.data = defaultdict(lambda: dict(calls=0, seconds=0.0, bytes=0, peak=0))
        self.phase = "setup"
        self.events = []
        self.lock = threading.Lock()

    def wrap(self, owner, name, label, size=None):
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapper(*a, **kw):
            phase = self.phase
            start = time.perf_counter()
            try:
                result = original(*a, **kw)
                if size:
                    with self.lock:
                        self.data[phase + ":" + label]["bytes"] += size(a, kw, result)
                return result
            finally:
                with self.lock:
                    m = self.data[phase + ":" + label]
                    m["calls"] += 1
                    m["seconds"] += time.perf_counter() - start

        setattr(owner, name, wrapper)

    @contextlib.contextmanager
    def span(self, name):
        self.phase = name
        gc.collect()
        before = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
        if tracemalloc.is_tracing():
            tracemalloc.reset_peak()
        start = time.perf_counter()
        try:
            yield
        finally:
            event = dict(phase=name, seconds=time.perf_counter() - start)
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                event.update(python_retained=current, python_peak=peak)
                event.update(
                    python_before=before, python_retained_delta=current - before, python_peak_delta=peak - before
                )
            self.events.append(event)
            print(json.dumps({"event": event}), flush=True)


def worker(args):
    info = json.loads((ROOT / "fixtures" / (args.fixture + ".json")).read_text())
    metrics = Metrics()
    metrics.wrap(cs.CapturedSource, "validate", "validate")
    metrics.wrap(
        cs, "_native_entry", "native", lambda a, k, r: (r.size if hasattr(r, "size") else len(r[1])) if r else 0
    )
    metrics.wrap(cs, "_copy_objects", "copy_objects")
    metrics.wrap(cs, "_materialize", "materialize")
    metrics.wrap(cs, "_population", "population")
    metrics.wrap(immutable, "_inventory", "inventory")
    metrics.wrap(immutable, "_regular", "regular", lambda a, k, r: len(r))
    metrics.wrap(
        process,
        "_run_git",
        "git",
        lambda a, k, r: len(r.stdout) if isinstance(r.stdout, bytes) else len(r.stdout.encode()),
    )
    # Count streamed blob yields separately; generator timing is inclusive of consumers.
    if hasattr(cs, "_BlobReader"):
        read = cs._BlobReader.read

        def counted_read(self, oid, consume):
            def counted_consume(chunk):
                with metrics.lock:
                    metrics.data[metrics.phase + ":blobs"]["bytes"] += len(chunk)
                return consume(chunk)

            return read(self, oid, counted_consume)

        cs._BlobReader.read = counted_read
    else:
        old_blobs = cs._blobs

        def counted_blobs(*a, **kw):
            for oid, body in old_blobs(*a, **kw):
                metrics.data[metrics.phase + ":blobs"]["bytes"] += len(body)
                yield oid, body

        cs._blobs = counted_blobs
    launched = subprocess.Popen

    def counted_launch(*a, **kw):
        with metrics.lock:
            metrics.data[metrics.phase + ":processes"]["calls"] += 1
            if list(a[0])[-2:] == ["cat-file", "--batch"]:
                metrics.data[metrics.phase + ":raw_readers"]["calls"] += 1
        return launched(*a, **kw)

    subprocess.Popen = counted_launch
    if args.trace:
        tracemalloc.start()
    dest = Path(args.destination).resolve()
    captured = None
    error = None
    factories = []
    try:
        with metrics.span("capture"):
            captured = cs.capture_source(
                Path(info["repo"]), dest, base_sha=info["base"], head_sha=info["head"], local=True
            )
        disk_capture = disk_stats(dest)
        objects = disk_stats(dest / ".git/objects")
        with metrics.span("validate"):
            captured.validate()
        if args.scopes:
            from daydream.deep.detection import StackAssignment
            from daydream.hunk_index import write_hunk_index
            from daydream.review_stage_inputs import StageInputFactory
            from daydream.workspace import WorkContext
            from tests.harness.backend import ScriptedBackend

            generated = dest / ".daydream"
            generated.mkdir(exist_ok=True)
            diff_path = generated / "diff.patch"
            diff_path.write_text(captured.diff)
            write_hunk_index(generated, captured.diff)
            work = WorkContext(
                repo=dest,
                source=Path(info["repo"]),
                base_branch="base",
                base_sha=info["base"],
                head_branch=None,
                head_sha=info["head"],
                is_ephemeral=True,
                run_id="offline",
                captured_source=captured,
            )
            revision = {"head_sha": info["head"], "merge_base_sha": info["base"], "source_id": captured.source_id}
            with metrics.span("planned_factories"):
                for i in range(args.scopes):
                    factories.append(
                        StageInputFactory(
                            ScriptedBackend(),
                            work,
                            StackAssignment("python#" + str(i), list(captured.files)),
                            diff_path=diff_path,
                            hunk_index_path=generated / "hunk-index.json",
                            shared_paths={},
                            revision=revision,
                            artifact_session=None,
                            allow_standalone=True,
                            read_only=True,
                        )
                    )
            installed = []

            def admission(i):
                factory = factories[i % len(factories)]

                def resolve(canonicalize):
                    for part in factory.parts[: args.targets]:
                        assignment = part.assignment
                        selector = {
                            "source_id": assignment["source_options"][0]["source_id"],
                            "basis": "supplied_diff",
                            "source_range": None,
                        }
                        canonicalize([selector], assignment["source_refs"])

                if hasattr(factory, "admit_submission"):

                    def prepare(sources):
                        resolve(sources.canonicalize)
                        return lambda: installed.append(i)

                    factory.admit_submission(prepare)
                else:
                    factory.canonicalize_source_selectors([], [], require_assigned=False)
                    resolve(factory.canonicalize_source_selectors)
                    factory.canonicalize_source_selectors([], [], require_assigned=False)
                    installed.append(i)

            with metrics.span("admissions"):
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    list(pool.map(admission, range(args.admissions)))
            with metrics.span("release_factories"):
                factories.clear()
                gc.collect()
        with metrics.span("cleanup"):
            captured.remove()
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        disk_capture = disk_stats(dest)
        objects = disk_stats(dest / ".git/objects")
        if captured is not None:
            with metrics.span("failed_cleanup"):
                try:
                    captured.remove()
                except BaseException as cleanup:
                    error["cleanup"] = {"type": type(cleanup).__name__, "message": str(cleanup)}
    result = dict(
        fixture=info,
        trace=args.trace,
        scopes=args.scopes,
        targets=args.targets,
        admissions=args.admissions,
        concurrency=args.concurrency,
        phases=metrics.events,
        counters=dict(metrics.data),
        disk_capture=disk_capture,
        objects=objects,
        error=error,
        destination_exists=dest.exists(),
        diff_bytes=None if captured is None else len(captured.diff.encode()),
        changed_targets=None if captured is None else len(captured.files),
        source_id=None if captured is None else captured.source_id,
        diff_sha256=None if captured is None else hashlib.sha256(captured.diff.encode()).hexdigest(),
        inventories_sha256=None
        if captured is None
        else hashlib.sha256(json.dumps([captured._before, captured._after]).encode()).hexdigest(),
    )
    result["implementation"] = {
        "code": str(CHECKOUT),
        "python": sys.version,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "owners": {
            name: hashlib.sha256((CHECKOUT / name).read_bytes()).hexdigest()
            for name in (
                "daydream/git_ops/captured_source.py",
                "daydream/git_ops/immutable.py",
                "daydream/git_ops/process.py",
                "daydream/review_stage_inputs.py",
                "daydream/review_source_refs.py",
                "daydream/review_investigation.py",
                "daydream/committed_review.py",
            )
        },
    }
    if tracemalloc.is_tracing():
        result["top_retained"] = [str(x) for x in tracemalloc.take_snapshot().statistics("lineno")[:12]]
    print(json.dumps({"result": result}), flush=True)


def sample_run(params, repeat):
    import ctypes

    import psutil

    class RUsage(ctypes.Structure):
        # sys/resource.h RUSAGE_INFO_V0, SDK shipped with Xcode on this machine.
        _fields_ = [("uuid", ctypes.c_ubyte * 16)] + [
            (k, ctypes.c_uint64)
            for k in (
                "user",
                "system",
                "idle",
                "interrupt",
                "pageins",
                "wired",
                "resident",
                "footprint",
                "start",
                "exit",
            )
        ]

    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    tag = (
        "-".join(
            str(params.get(k, v))
            for k, v in [("fixture", ""), ("scopes", 0), ("targets", 1), ("admissions", 0), ("concurrency", 1)]
        )
        + "-"
        + ("trace" if params.get("trace") else "time")
        + f"-{repeat}"
    )
    dest = runs / (tag + "-capture")
    if dest.exists():
        raise SystemExit("refusing to replace an existing capture: " + str(dest))
    argv = [sys.executable, str(Path(__file__).resolve()), "worker", "--destination", str(dest)]
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                argv.append("--" + key)
        else:
            argv.extend(["--" + key, str(value)])
    log = ROOT / "raw" / (tag + ".jsonl")
    log.parent.mkdir(exist_ok=True)
    if log.exists():
        raise SystemExit("refusing to overwrite an existing run: " + str(log))
    samples = []
    with log.open("w") as out:
        proc = subprocess.Popen(argv, cwd=CHECKOUT, stdout=out, stderr=subprocess.PIPE)
        parent = psutil.Process(proc.pid)
        start = time.perf_counter()
        peak_disk = 0
        while proc.poll() is None:
            rss = footprint = 0
            pids = []
            try:
                procs = [parent, *parent.children(recursive=True)]
            except psutil.Error:
                procs = []
            for p in procs:
                try:
                    rss += p.memory_info().rss
                    pids.append(p.pid)
                    usage = RUsage()
                    if libproc.proc_pid_rusage(p.pid, 0, ctypes.byref(usage)) == 0:
                        footprint += usage.footprint
                except psutil.Error:
                    pass
            if len(samples) % 10 == 0:
                peak_disk = max(peak_disk, disk_stats(dest)["logical"])
            samples.append(dict(t=time.perf_counter() - start, rss_sum=rss, phys_footprint_sum=footprint, pids=pids))
            time.sleep(0.01)
        stderr = proc.stderr.read().decode()
        code = proc.wait()
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    result = next((x["result"] for x in rows if "result" in x), None)
    if result:
        result.update(
            tag=tag,
            repeat=repeat,
            argv=argv,
            returncode=code,
            stderr=stderr,
            tree_rss_sum_peak=max((x["rss_sum"] for x in samples), default=0),
            tree_phys_footprint_sum_peak=max((x["phys_footprint_sum"] for x in samples), default=0),
            uss_unavailable="macOS task_for_pid denies cross-process psutil.memory_full_info; not reported",
            sampled_disk_logical_peak=peak_disk,
            sample_interval_seconds=0.01,
        )
        peak_disk = max(peak_disk, result["disk_capture"]["logical"])
        result["disk_logical_peak_boundaries_or_sampled"] = peak_disk
        with (ROOT / "results.jsonl").open("a") as f:
            f.write(json.dumps(result) + "\n")
    (log.with_suffix(".samples.json")).write_text(json.dumps(samples))
    if not result:
        print("FAILED", tag, code, stderr, flush=True)
    else:
        print(
            tag,
            "ERROR=" + str(result["error"])
            if result["error"]
            else " ".join(f"{x['phase']}={x['seconds']:.3f}s" for x in result["phases"]),
            flush=True,
        )
    # Leave logs; dispose resource-heavy run fixtures including competing bytes
    # only after recording proof. This is the harness's own disposable directory.
    for path in (dest,):
        if path.exists():
            shutil.rmtree(path)
    return result


def recorded_fixture(source, head, base):
    """Pin raw tracked bytes and the full physical store without mutating the source."""
    source = source.resolve(strict=True)
    if ROOT.is_relative_to(source):
        raise SystemExit("benchmark output must stay outside the recorded source")
    head = run_git(source, "rev-parse", "--verify", head + "^{commit}").decode().strip()
    base = run_git(source, "rev-parse", "--verify", base + "^{commit}").decode().strip()
    repo = ROOT / "fixtures" / "real-head"
    if repo.exists():
        raise SystemExit("recorded fixture must be absent")
    repo.mkdir(parents=True)
    run_git(repo, "init", "--template=", ".")
    common = Path(os.fsdecode(run_git(source, "rev-parse", "--git-common-dir")).strip())
    if not common.is_absolute():
        common = source / common
    shutil.copytree(common / "objects", repo / ".git/objects", dirs_exist_ok=True)
    rows = run_git(source, "ls-tree", "-r", "-z", head).split(b"\0")
    with subprocess.Popen(
        ["git", "cat-file", "--batch"], cwd=source, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        env=git_environment(),
    ) as child:
        try:
            for row in rows:
                if not row:
                    continue
                metadata, rawpath = row.split(b"\t", 1)
                mode, kind, oid = metadata.split()
                if mode == b"160000":
                    (repo / os.fsdecode(rawpath)).mkdir(parents=True, exist_ok=True)
                    continue
                child.stdin.write(oid + b"\n")
                child.stdin.flush()
                actual, actual_kind, size = child.stdout.readline().split()
                if (actual, actual_kind) != (oid, b"blob"):
                    raise RuntimeError("recorded object unavailable")
                body = child.stdout.read(int(size))
                if len(body) != int(size) or child.stdout.read(1) != b"\n":
                    raise RuntimeError("truncated object")
                target = repo / os.fsdecode(rawpath)
                target.parent.mkdir(parents=True, exist_ok=True)
                if mode == b"120000":
                    target.symlink_to(os.fsdecode(body))
                else:
                    target.write_bytes(body)
                    target.chmod(0o755 if mode == b"100755" else 0o644)
            child.stdin.close()
            if child.wait(timeout=30):
                raise RuntimeError("recorded object reader failed")
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
    (repo / ".git/HEAD").write_text(head + "\n")
    run_git(repo, "read-tree", head)
    info = dict(
        name="real-head", repo=str(repo), head=head, base=base, source_objects=disk_stats(repo / ".git/objects")
    )
    (repo.parent / "real-head.json").write_text(json.dumps(info, indent=2))


def filesystem_digest(root):
    """Independent exact source bytes/modes, including index, refs and the physical store."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        digest.update(json.dumps([path.relative_to(root).as_posix(), info.st_mode]).encode())
        if path.is_symlink():
            digest.update(os.fsencode(os.readlink(path)))
        elif path.is_file():
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


def integrity(name):
    """Full readable old/new correspondence and source immutability, outside timing runs."""
    from daydream.git_ops.models import PathAbsentError
    from daydream.git_ops.source import FrozenSourceRejected

    info = json.loads((ROOT / "fixtures" / (name + ".json")).read_text())
    source = Path(info["repo"])
    before = filesystem_digest(source)
    with tempfile.TemporaryDirectory(prefix="integrity-", dir=ROOT) as raw:
        captured = cs.capture_source(
            source, Path(raw) / "capture", base_sha=info["base"], head_sha=info["head"], local=True
        )
        captured.validate()
        reads = []
        paths = sorted({name for _, _, name in (*captured._before, *captured._after)})
        for side in ("old", "new"):
            for path in paths:
                try:
                    oid, body = captured.read(side, path)
                    reads.append([side, path, oid, len(body), hashlib.sha256(body).hexdigest()])
                except (PathAbsentError, FrozenSourceRejected) as exc:
                    reads.append([side, path, type(exc).__name__])
        # Large text lookup is deliberately unavailable; native bytes still get
        # their own streaming digest and full Git-OID verification above.
        native_digest = hashlib.sha256()
        for mode, oid, path in captured._after:
            if mode not in {"100644", "100755"}:
                continue
            native_digest.update(json.dumps([mode, path]).encode())
            with (captured.root / path).open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    native_digest.update(chunk)
        proof = dict(
            source_id=captured.source_id,
            head_sha=captured.head_sha,
            diff_sha256=hashlib.sha256(captured.diff.encode()).hexdigest(),
            targets=list(captured.files),
            inventories_sha256=hashlib.sha256(json.dumps([captured._before, captured._after]).encode()).hexdigest(),
            reads=reads,
            native_sha256=native_digest.hexdigest(),
            source_before=before,
            source_after=filesystem_digest(source),
        )
        if proof["source_before"] != proof["source_after"]:
            raise RuntimeError("source mutated")
        captured.remove()
    target = ROOT / (name + "-integrity.json")
    if target.exists():
        raise SystemExit("refusing to overwrite proof")
    target.write_text(json.dumps(proof, indent=2))
    print(target)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--fixture", required=True)
    worker_parser.add_argument("--destination", required=True)
    worker_parser.add_argument("--trace", action="store_true")
    for name, default in [("scopes", 0), ("targets", 1), ("admissions", 0), ("concurrency", 1)]:
        worker_parser.add_argument("--" + name, type=int, default=default)
    sub.add_parser("fixtures")
    execute = sub.add_parser("run")
    execute.add_argument("--fixture", required=True)
    execute.add_argument("--trace", action="store_true")
    execute.add_argument("--repeat", type=int, default=3)
    for name, default in [("scopes", 0), ("targets", 1), ("admissions", 0), ("concurrency", 1)]:
        execute.add_argument("--" + name, type=int, default=default)
    record = sub.add_parser("recorded-fixture")
    record.add_argument("--source", type=Path, required=True)
    record.add_argument("--head", required=True)
    record.add_argument("--base", required=True)
    proof = sub.add_parser("integrity")
    proof.add_argument("--fixture", required=True)
    args = parser.parse_args()
    if args.command == "worker":
        worker(args)
    elif args.command == "fixtures":
        env = {
            "revision": run_git(CHECKOUT, "rev-parse", "HEAD").decode().strip(),
            "python": sys.version,
            "git": run_git(CHECKOUT, "--version").decode().strip(),
            "platform": platform.platform(),
            "cpu": os.cpu_count(),
            "ram_bytes": subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip(),
            "filesystem": subprocess.check_output(["df", "-h", str(ROOT)]).decode(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (ROOT / "environment.json").write_text(json.dumps(env, indent=2))
        for size in (0, 32, 128):
            for layout in ("loose", "packed"):
                fixture(f"h{size}-{layout}", history_mib=size, layout=layout)
        for mib in (1, 16, 64, 240):
            fixture(f"bytes-{mib}", source_bytes=mib * 1024**2)
        for count in (16, 256, 1024):
            fixture(f"files-{count}", files=count)
        for changes in (1, 8, 32):
            fixture(f"targets-{changes}", files=64, changes=changes)
        for lines in (1, 1024, 8192):
            fixture(f"patch-{lines}", files=32, patch_lines=lines)
        fixture("over-store", oversized_store=True)
        fixture("over-source", oversized_source=True)
        fixture("large-leaf", source_bytes=64 * 1024**2, files=2)
        print("fixtures ready", flush=True)
    elif args.command == "recorded-fixture":
        recorded_fixture(args.source, args.head, args.base)
    elif args.command == "integrity":
        integrity(args.fixture)
    else:
        params = vars(args).copy()
        params.pop("command")
        repeat = params.pop("repeat")
        for i in range(repeat):
            result = sample_run(params, i)
            if result is None or result["error"]:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
