#!/usr/bin/env python3
"""
Prepare an FFmpeg control-flow-graph (CFG) dataset for GMN experiments.

Pipeline:
  1. Build FFmpeg 4.1.4 with GCC and Clang at O0-O3.
  2. Recover function-level CFGs with angr CFGFast.
  3. Store one CFG per (function, compiler, optimization) in SQLite.
  4. Keep functions appearing in at least --min-variants builds.
  5. Create an 80/10/10 split by FUNCTION NAME (not by CFG instance).

This intentionally keeps compiler/optimization metadata and assembly
instructions so the data can be used for both structure-only and
assembly-aware GMN experiments.

Note:
The original GMN paper specifies GCC + Clang and O0-O3, but does not
specify exact compiler versions in the main paper. Therefore, the default
setup below uses the compilers supplied by your machine. Exact numeric
reproduction of the paper's FFmpeg benchmark is not guaranteed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Iterable

FFMPEG_VERSION = "4.1.4"
VARIANTS = [
    ("gcc", "O0"),
    ("gcc", "O1"),
    ("gcc", "O2"),
    ("gcc", "O3"),
    ("clang", "O0"),
    ("clang", "O1"),
    ("clang", "O2"),
    ("clang", "O3"),
]


def run(
    cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None
) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def check_program(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required program not found in PATH: {name}")


def compiler_version(compiler: str) -> str:
    try:
        out = subprocess.check_output(
            [compiler, "--version"], text=True, stderr=subprocess.STDOUT
        )
        return out.splitlines()[0].strip()
    except Exception:
        return "unknown"


def build_variant(
    source_dir: Path,
    output_dir: Path,
    compiler: str,
    opt: str,
    jobs: int,
    configure_args: list[str],
) -> Path:
    variant = f"{compiler}_{opt}"
    build_dir = output_dir / "builds" / variant
    binary_dir = output_dir / "binaries" / variant
    build_dir.parent.mkdir(parents=True, exist_ok=True)
    binary_dir.mkdir(parents=True, exist_ok=True)

    if build_dir.exists():
        shutil.rmtree(build_dir)
    shutil.copytree(source_dir, build_dir)

    env = os.environ.copy()
    env["CC"] = compiler
    if shutil.which("clang++") and compiler == "clang":
        env["CXX"] = "clang++"
    env["CFLAGS"] = f"-{opt} -g"
    env["CXXFLAGS"] = f"-{opt} -g"

    config = [
        "./configure",
        "--disable-doc",
        "--disable-ffplay",
        "--disable-ffprobe",
        "--disable-stripping",
        "--disable-x86asm",
        *configure_args,
    ]

    run(config, cwd=build_dir, env=env)
    run(["make", f"-j{jobs}", "ffmpeg"], cwd=build_dir, env=env)

    src_binary = build_dir / "ffmpeg"
    if not src_binary.exists():
        raise FileNotFoundError(f"FFmpeg binary not found: {src_binary}")

    dst_binary = binary_dir / "ffmpeg"
    shutil.copy2(src_binary, dst_binary)
    return dst_binary


def build_all(
    source_dir: Path,
    output_dir: Path,
    gcc: str,
    clang: str,
    jobs: int,
    configure_args: list[str],
) -> None:
    for compiler_name, opt in VARIANTS:
        compiler = gcc if compiler_name == "gcc" else clang
        check_program(compiler)
        print(f"\n=== Building {compiler_name} {opt} ===")
        binary = build_variant(
            source_dir=source_dir,
            output_dir=output_dir,
            compiler=compiler,
            opt=opt,
            jobs=jobs,
            configure_args=configure_args,
        )
        print(f"Saved: {binary}")

    meta = {
        "ffmpeg_version": FFMPEG_VERSION,
        "variants": [
            {
                "compiler": c,
                "optimization": o,
                "compiler_version": compiler_version(gcc if c == "gcc" else clang),
            }
            for c, o in VARIANTS
        ],
    }
    (output_dir / "build_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def get_instruction_records(project, block_addr: int) -> list[dict]:
    try:
        block = project.factory.block(block_addr)
        records = []
        for insn in block.capstone.insns:
            records.append(
                {
                    "address": int(insn.address),
                    "mnemonic": insn.mnemonic,
                    "op_str": insn.op_str,
                }
            )
        return records
    except Exception:
        return []


def extract_binary(
    binary: Path,
    db_path: Path,
    compiler: str,
    opt: str,
    include_instructions: bool = True,
) -> int:
    try:
        import angr
    except ImportError as exc:
        raise RuntimeError("angr is required. Install with: pip install angr") from exc

    print(f"Analyzing {binary}")
    project = angr.Project(
        str(binary),
        auto_load_libs=False,
        load_options={"auto_load_libs": False},
    )

    cfg = project.analyses.CFGFast(
        normalize=False,
        show_progressbar=True,
        data_references=False,
        cross_references=False,
    )

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS graphs (
            graph_id TEXT PRIMARY KEY,
            function_name TEXT NOT NULL,
            compiler TEXT NOT NULL,
            optimization TEXT NOT NULL,
            variant TEXT NOT NULL,
            num_nodes INTEGER NOT NULL,
            num_edges INTEGER NOT NULL,
            graph_blob BLOB NOT NULL
        )
        """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_function ON graphs(function_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_variant ON graphs(variant)")

    count = 0
    variant = f"{compiler}_{opt}"

    for function in cfg.kb.functions.values():
        if function.is_plt:
            continue

        if function.size < 32:
            continue

        if len(function.block_addrs) < 3:
            continue
        
        # Skip PLT/import stubs; these are not source functions from FFmpeg.
        if getattr(function, "is_plt", False):
            continue

        name = str(function.name).strip()
        if not name:
            continue

        block_addrs = sorted(int(a) for a in function.block_addrs)
        if not block_addrs:
            continue

        allowed = set(block_addrs)
        edges = []

        try:
            graph = function.transition_graph
            for src, dst in graph.edges():
                s = int(src.addr)
                d = int(dst.addr)
                if s in allowed and d in allowed:
                    edges.append([s, d])
        except Exception:
            continue

        nodes = []
        for addr in block_addrs:
            node = {"id": addr}
            if include_instructions:
                node["instructions"] = get_instruction_records(project, addr)
            nodes.append(node)

        record = {
            "function_name": name,
            "compiler": compiler,
            "optimization": opt,
            "variant": variant,
            "entry": int(function.addr),
            "nodes": nodes,
            "edges": edges,
        }

        key = f"{name}|{variant}".encode("utf-8", "replace")
        graph_id = hashlib.sha1(key).hexdigest()

        blob = gzip.compress(
            json.dumps(record, separators=(",", ":")).encode("utf-8"),
            compresslevel=6,
        )

        conn.execute(
            """
            INSERT OR REPLACE INTO graphs
            (graph_id, function_name, compiler, optimization, variant,
             num_nodes, num_edges, graph_blob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                graph_id,
                name,
                compiler,
                opt,
                variant,
                len(nodes),
                len(edges),
                sqlite3.Binary(blob),
            ),
        )

        count += 1

    conn.commit()
    conn.close()
    return count


def extract_all(output_dir: Path, no_instructions: bool) -> None:
    db_path = output_dir / "ffmpeg_cfg.sqlite"
    binaries = output_dir / "binaries"

    counts = {}

    for compiler, opt in VARIANTS:
        variant = f"{compiler}_{opt}"
        binary = binaries / variant / "ffmpeg"
        if not binary.exists():
            print(f"Skipping missing binary: {binary}")
            continue

        count = extract_binary(
            binary=binary,
            db_path=db_path,
            compiler=compiler,
            opt=opt,
            include_instructions=not no_instructions,
        )
        counts[variant] = count
        print(f"{variant}: {count} functions")

    (output_dir / "extraction_counts.json").write_text(
        json.dumps(counts, indent=2), encoding="utf-8"
    )


def create_split(output_dir: Path, seed: int, min_variants: int) -> None:
    import random

    db_path = output_dir / "ffmpeg_cfg.sqlite"
    conn = sqlite3.connect(db_path)

    rows = conn.execute(
        "SELECT function_name, COUNT(DISTINCT variant) FROM graphs GROUP BY function_name"
    ).fetchall()
    conn.close()

    eligible = sorted(name for name, count in rows if int(count) >= min_variants)

    rng = random.Random(seed)
    rng.shuffle(eligible)

    n = len(eligible)
    n_train = int(0.80 * n)
    n_val = int(0.10 * n)

    split = {
        "seed": seed,
        "min_variants": min_variants,
        "num_functions": n,
        "train": eligible[:n_train],
        "val": eligible[n_train : n_train + n_val],
        "test": eligible[n_train + n_val :],
    }

    (output_dir / "splits.json").write_text(
        json.dumps(split, indent=2), encoding="utf-8"
    )

    print(
        f"Split: {len(split['train'])} train / "
        f"{len(split['val'])} val / {len(split['test'])} test functions"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build/extract FFmpeg CFG dataset for GMN"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="Build 8 FFmpeg binaries")
    b.add_argument("--source-dir", type=Path, required=True)
    b.add_argument("--out", type=Path, default=Path("data/ffmpeg"))
    b.add_argument("--gcc", default="gcc")
    b.add_argument("--clang", default="clang")
    b.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    b.add_argument("--configure-arg", action="append", default=[])

    e = sub.add_parser("extract", help="Extract CFGs from existing binaries")
    e.add_argument("--out", type=Path, default=Path("data/ffmpeg"))
    e.add_argument("--no-instructions", action="store_true")

    s = sub.add_parser("split", help="Create function-level train/val/test split")
    s.add_argument("--out", type=Path, default=Path("data/ffmpeg"))
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--min-variants", type=int, default=2)

    a = sub.add_parser("all", help="Build + extract + split")
    a.add_argument("--source-dir", type=Path, required=True)
    a.add_argument("--out", type=Path, default=Path("data/ffmpeg"))
    a.add_argument("--gcc", default="gcc")
    a.add_argument("--clang", default="clang")
    a.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a.add_argument("--configure-arg", action="append", default=[])
    a.add_argument("--no-instructions", action="store_true")
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--min-variants", type=int, default=2)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.command == "build":
        build_all(
            args.source_dir.resolve(),
            args.out.resolve(),
            args.gcc,
            args.clang,
            args.jobs,
            args.configure_arg,
        )

    elif args.command == "extract":
        extract_all(args.out.resolve(), args.no_instructions)

    elif args.command == "split":
        create_split(args.out.resolve(), args.seed, args.min_variants)

    elif args.command == "all":
        build_all(
            args.source_dir.resolve(),
            args.out.resolve(),
            args.gcc,
            args.clang,
            args.jobs,
            args.configure_arg,
        )
        extract_all(args.out.resolve(), args.no_instructions)
        create_split(args.out.resolve(), args.seed, args.min_variants)


if __name__ == "__main__":
    main()
