# Wormhole N150, measured

The `dual-noc-mesh` topology (8×8 grid, one L1 per core, NoC0 loads/gathers, NoC1 stores) at 1 GHz, with matmul, DRAM
and multicast rates from n150 measurements and tt-metal reports. `tests/wormhole` keeps the uncalibrated fixture rates.

## Units

Cost = `latency + volume/throughput` cycles. Movers carry `effective_bandwidth = 10·f`, where f = transfers sharing the
path (8 for a full-mesh unicast, 1 for a row/column multicast). A DRAM throughput `t` is therefore `t/10` B/cycle per
lane with 8 lanes, and `0.8·t` B/cycle aggregate.

## Rates vs the fixture

| Knob | fixture | here | Source |
|---|---|---|---|
| matmul (per call) | `M*N/2 + 2MNK/716` | `684 + MN/13.4 + 2K(M+N)/27.7 + 2MNK/1773`; batched: all × B | n150 single-core HiFi2 fit, 32×32..256×256 blocks (319..1172 FLOP/cycle) |
| DRAM→L1 unicast | 150 (120 B/cyc aggr.) | 312.5 (250 B/cyc aggr.) | tt-metal `Saturating_DRAM_bandwidth`: LLM decode 239–261 GB/s @12 Gbps |
| DRAM→L1 multicast | 150 | 200 (20 B/cyc per row) | NoC0 row multicast 161.7 GB/s over 8 rows (measured) |
| L1→DRAM | `M*N*2`, 150, no contention | `M*N*2*effective_bandwidth`, 312.5 | reads and writes share the DRAM banks |
| L1→L1 gather | 28 (2.8 B/cyc) | 320 (32 B/cyc) | 32-B NoC links; tt-metal CI core-to-core 28–31 B/cyc |
| Usable L1/core | 1,398,784 B | 1,393,152 B | tt-metal `6024ed4` 1,393,312 B, rounded down to word × banks |

Kept from the fixture, unverified: DRAM latency 454, multicast/gather latency 344. HiFi4 matmul alternative (n150 fit):
`718 + MN/14 + 2K(M+N)/29 + 2MNK/976`.

## Limits

- 8 modelled DRAM lanes vs 12 physical banks: only the aggregate rate matches (one unicast stream gets 31 B/cyc; one bank
  delivers ~22).
- HiFi2 matmul everywhere. tt-metal uses HiFi4 for bf16-KV SDPA and SDPA prefill.
- The matmul fit is one core with inputs in L1; every K-block of a core's loop is one call.
- Vector rates (sources in `vector_lane.yaml` comments): add/sub/sum are FPU rates, mul is derived for HiFi2, and the SFPU
  ops are n150 single-tile microbenchmarks (75 cycles pack/sync per call + per-tile cost).
- No NoC routing or link contention beyond f.

## Use with Loom

```bash
cargo build --release -p mlar-frontend --bin eval_runtime --bin export_platform
target/release/export_platform frontend/examples/declarative/wormhole-n150 <out>/wormhole_n150.mlir   # hw_spec
export LOOM_ARCH_DIR=$PWD/frontend/examples/declarative/wormhole-n150 LOOM_EVAL_SYSTEM=$PWD/target/release/eval_runtime
```
