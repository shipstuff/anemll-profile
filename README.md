# anemll-profile

<img src="assets/anemll-profile-icon.png" alt="ANEMLL Profiler Icon" width="240">

ANE (Apple Neural Engine) profiling toolkit for Core ML models. Two complementary
tools:

- **`anemll-profile`** — runtime-measured profiler: per-op cost estimates,
  device placement, actual prediction throughput, graph interruptions,
  bandwidth and memory/compute bound classification.
- **`ane-costplan`** — compile-time cost analyzer: per-op cost weights from
  Apple's `MLComputePlan` API (the ANE compiler's internal cost model), with
  device support/preference per operation.

Use both together: `anemll-profile` shows what *actually* runs slow,
`ane-costplan` shows what the compiler *thinks* is expensive. Disagreements
between the two reveal optimization opportunities.

For a compact agent-oriented workflow, see [AGENTS.md](./AGENTS.md).

> This fork adds [`ane-costplan`](./ane_costplan.swift) — a compile-time cost
> analyzer using Apple's `MLComputePlan` API.

## Install

Build both binaries from source:

```bash
git clone https://github.com/shipstuff/anemll-profile.git
cd anemll-profile
make           # builds anemll-profile and ane-costplan
sudo make install
```

Verify:

```bash
anemll-profile --version
ane-costplan --help 2>&1 | head -3
```

Build from source is the only way to get both tools together. (Upstream
homebrew tap installs `anemll-profile` only.)

## Usage

### `anemll-profile` — measured runtime profiler

```bash
anemll-profile model.mlmodelc
anemll-profile model.mlpackage
anemll-profile /path/to/model          # auto-detects .mlmodelc or .mlpackage
anemll-profile -a model.mlmodelc       # include GPU in device assignment
anemll-profile --interrupt-ms 150 model.mlmodelc   # change heuristic ANE boundary cost
anemll-profile -j report.json model.mlmodelc       # write structured JSON report
```

Reports:
- **Op-Type Runtime Breakdown** — per-op-type estimated runtime, GFLOP/s, GB/s, memory/compute bound
- **ANE Graph Interruptions** — non-ANE islands that interrupt ANE execution, ranked by estimated latency tax
- **Measured Prediction** — actual wall-clock time, iter/s, weight bandwidth GB/s
- **Top Expensive Ops** — the 20 slowest operations
- **Conv Detail** — convolution ops with channel counts and work unit efficiency
- **CPU/GPU Fallback** — ops not on ANE with specific compiler reasons

### `ane-costplan` — compile-time cost analyzer *(new in this fork)*

```bash
ane-costplan model.mlmodelc
ane-costplan -j report.json model.mlmodelc     # structured JSON output
```

Reports:
- **Per-op cost weights** from `MLComputePlan.estimatedCost(of:)` — Apple's own
  cost model that drives ANE placement and scheduling decisions
- **Device support/preference** per op — whether ANE, GPU, or CPU is preferred
  and what's supported
- **Top 10 by cost** — highlights the compiler's view of the expensive ops

### When to use each

| Question | Tool |
|---|---|
| How fast does this model actually run? | `anemll-profile` |
| What bandwidth is each op hitting? | `anemll-profile` |
| Why did these ops fall back to CPU? | `anemll-profile` |
| What does the compiler think is expensive? | `ane-costplan` |
| Does placement match my expectations per-op? | `ane-costplan` |
| Will a graph change reduce compiler cost? | `ane-costplan` (cheap, no prediction needed) |
| End-to-end optimization: find mismatches | both |

`ane-costplan` is especially useful as a fast feedback loop — it returns in
seconds (no prediction run), so you can iterate on graph modifications and
see the cost model's response before paying the full compile + profile cycle.

## How `anemll-profile` works

1. Loads `MLComputePlan` to get per-op device assignment and cost weights
2. Captures Espresso `[CostModelFeature]` logs via forked `/usr/bin/log stream`
3. Parses `Unsupported op` compiler messages for ANE fallback reasons
4. Analyzes ordered MLComputePlan ops to find ANE graph interruption islands, including CPU or GPU detours
5. Applies a heuristic ANE boundary penalty (300 ms by default) to rank interruption hot spots
6. Runs actual predictions with dummy inputs to measure real throughput
7. Computes weight-only DRAM bandwidth (excludes L2-resident activations)

## How `ane-costplan` works

1. Loads `MLComputePlan` for the compiled model
2. Walks every operation in the MIL program
3. Queries `plan.estimatedCost(of: op)` for the compiler's cost weight per op
4. Queries `plan.deviceUsage(for: op)` for supported + preferred device per op
5. Sorts by cost weight descending to surface bottlenecks
6. Optionally emits structured JSON for agent workflows

No prediction runs, no log capture — pure compile-time analysis. Runs in seconds
on any compiled `.mlmodelc`.

## What's New (fork additions)

- **`ane-costplan` binary** — compile-time compiler cost analyzer (this fork)

What's new upstream (also in this fork):

- **Agent guide (`AGENTS.md`)** — first-class workflow guide for agents
- **ANE graph interruptions** — interruption analysis highlights non-ANE islands
- **Latency-ranked interruption hot spots** — rank by estimated switch penalty
- **Function timeline view** — compact accelerator timeline per function
- **Configurable switch heuristic** — `--interrupt-ms` / `--interrupt-boundary-ms`
- **Structured JSON export** — `-j` / `--json FILE`

## Requirements

- macOS 14+ (Sonoma) — requires `MLComputePlan` API
- Xcode Command Line Tools (for `clang` and `swiftc`)
- `Foundation` and `CoreML` system frameworks included with macOS

Recommended for agent automation:

- `jq` for parsing `-j` JSON reports

## License

MIT
