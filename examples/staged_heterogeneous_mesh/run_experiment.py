#!/usr/bin/env python3
"""Run the explicit-transfer staged-memory experiment end to end."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


MLAR_ROOT = Path(__file__).resolve().parents[2]
LOOM_DATAFLOW = MLAR_ROOT.parent / "loom-dataflow"
LOOM_ROOT = MLAR_ROOT.parents[1]
TOOLS = LOOM_DATAFLOW / "build/tool/loom-opt/single_stage"
WORKLOAD = Path(__file__).with_name("workload.mlir")


def run(command: list[str], *, stdout: Path | None = None) -> str:
    result = subprocess.run(command, cwd=MLAR_ROOT, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)}\n{result.stderr}")
    if stdout is not None:
        stdout.write_text(result.stdout)
    return result.stdout


def objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/loom-staged"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    hw = args.output / "hardware.mlir"
    mapped = args.output / "03_mapped.mlir"
    reuse = args.output / "04_reuse.mlir"
    explored = args.output / "05_broadcast.mlir"
    etg_path = args.output / "06_etg.json"
    resolved_path = args.output / "07_resolved.json"
    blocks_path = args.output / "08_blocks.json"
    materialized_path = args.output / "09_materialized.mlir"

    run(["cargo", "build", "--quiet", "--example", "staged_heterogeneous_mesh_evaluator"])
    hw.write_text(run(["cargo", "run", "--quiet", "--example", "staged_heterogeneous_mesh"]))
    run([str(TOOLS / "enumerate_hw_mapping"), "--input", str(WORKLOAD), "--hw_spec", str(hw)], stdout=mapped)
    run([str(TOOLS / "analyze_reuse"), "--input", str(mapped)], stdout=reuse)
    run([str(TOOLS / "enumerate_copy_broadcast"), "--input", str(reuse)], stdout=explored)
    run([str(TOOLS / "staged_etg"), "--input", str(explored), "--hw_spec", str(hw), "--output", str(etg_path)])

    sys.path.insert(0, str(LOOM_ROOT))
    os.environ["LOOM_EVAL_SYSTEM"] = str(
        MLAR_ROOT / "target/debug/examples/staged_heterogeneous_mesh_evaluator"
    )
    from loom.loom_utils.mlar import resolve_etg_variants

    variants = json.loads(etg_path.read_text())
    expected = {"load_gcram_f16", "load_rram_f16", "matmul_staged_f16_f32"}
    names = {obj["Func"]["func"]["name"] for variant in variants for obj in objects(variant) if "Func" in obj}
    if not all(any(name.startswith(prefix) for name in names) for prefix in expected):
        raise RuntimeError(f"ETG is missing staged operations: {sorted(names)}")
    for variant in variants:
        footprint = variant["constraint_scope"]["metadata"]["L1_footprint"]
        if footprint["capacity"] != 64 * 1024 or len(footprint["load"]) != 2:
            raise RuntimeError(f"unexpected staging footprint: {footprint}")

    resolved = resolve_etg_variants(variants, njobs=1)
    single_buffer = {"Eq": [{"Sym": "is_double_buffer"}, {"Const": 0}]}
    for variant in resolved:
        variant["constraint_scope"]["hard_constraints"].append(single_buffer)
    resolved_path.write_text(json.dumps(resolved, indent=2))

    from loom.solver.main import run as solve

    blocks = solve(
        resolved_path,
        njobs=1,
        symbol_domains={
            "tile_m": [32, 64, 96, 128],
            "tile_n": [32, 64, 96, 128],
            "tile_k": [32, 64],
        },
        topk_candidates=1,
    )
    blocks_path.write_text(json.dumps(blocks, indent=2))
    assignment = next(iter(blocks.values()))
    stage_bytes = 2 * (
        assignment["tile_m"] * assignment["tile_k"]
        + assignment["tile_k"] * assignment["tile_n"]
    )
    if assignment["is_double_buffer"] != 0 or stage_bytes > 64 * 1024:
        raise RuntimeError(f"invalid staging assignment: {assignment}, {stage_bytes} bytes")

    sys.path.insert(0, str(LOOM_DATAFLOW / "build/lib"))
    import _loom_pipeline

    error, materialized = _loom_pipeline.run_materialization_pipeline(
        explored.read_text(), json.dumps(blocks)
    )
    if error:
        raise RuntimeError(error)
    materialized_path.write_text(materialized)
    print(f"materialized {next(iter(blocks))}")
    print(f"block sizes: {assignment}; live staging: {stage_bytes} / {64 * 1024} bytes")
    print(f"artifacts: {args.output}")


if __name__ == "__main__":
    main()
