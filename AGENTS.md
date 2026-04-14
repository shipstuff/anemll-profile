# anemll-profile Agent Guide

Two ANE profiling tools, complementary:

- **`anemll-profile`** — measured runtime, bandwidth, graph interruptions
- **`ane-costplan`** — compile-time compiler cost weights (no prediction run)

## Install

```bash
git clone https://github.com/shipstuff/anemll-profile.git
cd anemll-profile
make && sudo make install
```

Verify:

```bash
anemll-profile --version
ane-costplan --help 2>&1 | head -3
```

Build from source gets both tools. (Upstream homebrew tap installs
`anemll-profile` only.)

## Dependencies

Required:
- macOS 14+ with the system `CoreML` and `Foundation` frameworks
- Xcode Command Line Tools for `xcrun clang`, `swiftc`, `make`, and SDK headers

Recommended for agent workflows:
- `jq` for reading `-j` JSON reports

No Python or pip dependencies.

## Which tool when

| Question | Tool | Why |
|---|---|---|
| How fast does it run? | `anemll-profile` | Runs predictions, measures wall-clock |
| What's the bandwidth per op? | `anemll-profile` | Reads live cost model features |
| Why did ops fall back? | `anemll-profile` | Captures Espresso compiler messages |
| What does the compiler score as expensive? | `ane-costplan` | Reads `MLComputePlan` cost weights |
| Is op X placed on ANE? | `ane-costplan` | `deviceUsage.preferredComputeDevice` |
| Did a graph change reduce compiler cost? | `ane-costplan` | No prediction needed, seconds to run |
| Fast iteration on graph optimizations | `ane-costplan` | ~2 s vs ~30 s for anemll-profile |
| End-to-end decision: ship it? | both | Cost model + measured perf should agree |

## Preferred Mode

Both tools support JSON output. Prefer it for agent workflows:

```bash
anemll-profile -j profile.json model.mlmodelc
ane-costplan -j costplan.json model.mlmodelc
```

Text output is good for humans; JSON is the source of truth for automation.

## `anemll-profile` — Fast Workflow

Focus on these JSON sections:
- `summary`
- `measured_prediction`
- `graph_interruptions`
- `fallback_ops`
- `op_type_breakdown`
- `top_operations`

Steps:
1. Run with `-j`.
2. Check `graph_interruptions`.
3. If interruptions are `0`, shift attention to `fallback_ops`, `op_type_breakdown`, `top_operations`.
4. If interruptions are nonzero, rank fixes by `estimated_total_tax_ms`.
5. Use `measured_prediction` to compare real impact after model changes.

### How to interpret `anemll-profile` results

- `graph_interruptions`: Non-ANE islands that break continuous ANE execution. Detour can be CPU or GPU.
- `fallback_ops`: Best place to see *why* specific ops didn't stay on ANE.
- `op_type_breakdown`: High-level memory-bound vs compute-bound hotspots.
- `top_operations`: Repeated expensive state updates, slices, LUT expansion, or large memory moves.

### Common optimization signals

- `Unsupported tensor data type: int32`: Reduce or isolate int32 control and indexing.
- Repeated `slice_update`, `slice_by_index`, or `read_state`: Investigate KV-cache layout.
- Large `constexpr_lut_to_dense` cost: LUT expansion may dominate memory bandwidth.
- Few CPU fallbacks with `0` interruptions: Model is mostly on ANE; focus on memory traffic.

## `ane-costplan` — Fast Workflow

Focus on these JSON sections:
- `total_cost`
- `total_ops`
- `with_device` / `no_device`
- `operations` (sorted descending by cost)

Steps:
1. Run with `-j`.
2. Check `no_device` — ops without a `MLComputeDevice` are CPU ops (gathers, int32 work).
3. Examine top operations by `cost` — these are what the compiler sees as expensive.
4. After a graph modification, re-run and diff `total_cost`. Drops indicate the compiler agrees with your change.

### How to interpret `ane-costplan` results

- `total_cost` is dimensionless, not milliseconds — compare relatively (before/after) not absolutely.
- `with_device` count = ops the compiler can place on ANE/GPU. `no_device` = CPU-only.
- The cost weights reflect *the compiler's scheduling decisions*, not runtime. A high-cost op may still pipeline well; a low-cost op on the critical path can still bottleneck.
- Use it to iterate on graph structure (fusion, layout, quantization choices) without paying full compile + predict costs.

### When `ane-costplan` disagrees with `anemll-profile`

This is a signal. Examples:
- High `ane-costplan` cost but low measured runtime → ANE is pipelining that op effectively
- Low `ane-costplan` cost but high measured runtime → probably a bandwidth or memory stall the static cost model doesn't capture
- Op has `preferred: ANE` in `ane-costplan` but `anemll-profile` shows it on CPU → check whether the op got placed off-ANE at runtime due to shape or tile constraints

## Model Directory Tip

If a folder contains multiple `.mlmodelc` or `.mlpackage` bundles, profile the
important submodels individually, e.g.:

- embeddings
- prefill
- decode or FFN body
- lm_head

Do not assume a parent directory is directly profileable unless it contains a
single model bundle. Both tools auto-detect `.mlmodelc` and `.mlpackage`.
