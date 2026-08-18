# Quickstart Validation: Memory-Centric Visualization

## Prerequisites

- A stable Rust toolchain supporting edition 2024.
- Node.js 20 or newer and npm.
- Run commands from the repository root.
- Chrome or Chromium is optional for visual checking; absence must be reported as a skipped check.

## 1. Install Adapter Dependencies

```bash
npm ci --prefix tools/mlar-archify
```

Expected outcome: installation completes from the committed lockfile.

## 2. Run Focused Adapter Tests

```bash
npm test --prefix tools/mlar-archify
```

Expected coverage:

- one combined memory hierarchy-and-access diagram is the default when it fits;
- recursive array/bank detail and unconnected memories remain visible;
- one processor reading and writing the same memory appears once with both edges;
- DRAM, L1, recursive banks, processors, and data movers share the same primary diagram;
- a DRAM-to-L1 data mover shows both endpoints and directions there;
- duplicate display names remain distinct through canonical IDs and scope paths;
- symbolic sizes retain expression text;
- hierarchy/access partitions never exceed 12 primary nodes;
- every source ID remains covered; and
- the gallery stays English-only and contains no inline architecture SVG.

## 3. Run Rust Visualization Regressions

```bash
cargo test visualization::document
cargo test --test visualization_export_test
cargo test --test 2d_mesh test_export_2d_mesh_torus_visualization_yaml
cargo fmt --check
```

Expected outcome: existing `mlar.visualization.v1` projection and reference-resolution tests pass without a schema or public API change.

The two focused export tests may rewrite `tests/2d_mesh/2d_mesh_torus.visualization.yaml`. Inspect its diff; under this plan it should have no semantic payload change.

## 4. Verify the Vendored Renderer

```bash
node tools/archify/bin/archify.mjs doctor
```

Expected outcome: the project-relative Archify installation reports usable validation and delivery capabilities.

## 5. Build the Representative Bundle

```bash
node tools/mlar-archify/bin/mlar-archify.mjs build \
  tests/2d_mesh/2d_mesh_torus.visualization.yaml \
  visualization-output/2d-mesh \
  --visual-check
```

Expected outcome:

- every generated specification passes Archify showcase validation and delivery;
- `visualization-output/2d-mesh/index.html` is generated;
- the default view is the unified root memory hierarchy-and-access diagram;
- DRAM and replicated L1 appear with scope/replication context;
- DRAM/L1 bank layers and all connected processors/data movers appear in that same diagram;
- `dram_l1_noc0` forms a visible DRAM → mover → L1 route;
- matrix and vector processors show read/write access at L1;
- network and resource relationships remain available in supporting views; and
- no view exceeds 12 primary nodes.

If browser automation is unavailable, the visual-check receipt must say skipped; do not describe it as passing.

## 6. Inspect Coverage and Receipts

```bash
jq -e '
  (.omitted_scope_ids | length) == 0 and
  (.omitted_component_ids | length) == 0 and
  (.omitted_relationship_ids | length) == 0 and
  .policy.maximum_primary_nodes == 12 and
  (.policy.expand_replicated_instances | not)
' visualization-output/2d-mesh/conversion-report.json

jq -e '
  all(.diagrams[];
    .validation.ok == true and
    .delivery.ok == true)
' visualization-output/2d-mesh/bundle-manifest.json
```

Expected outcome: both commands exit successfully. Also inspect generated specifications when verifying the exact primary-node count, because canonical coverage counts intentionally exclude presentation-only memory-layer context.

## 7. Perform the User Journeys

```bash
node tools/mlar-archify/bin/mlar-archify.mjs serve \
  visualization-output/2d-mesh
```

Open `http://127.0.0.1:4173/` and verify:

1. In the one primary diagram, identify DRAM, L1, their ownership levels, L1's 8×8 replication context, and both recursive bank layers without opening the source YAML.
2. Without changing diagrams, identify matrix and vector compute processors at L1 and confirm both read and write access.
3. In that same diagram, trace DRAM → `dram_l1_noc0` → L1, then trace L1 → `l1_dram_noc1` → DRAM.
4. Confirm that selecting hierarchy context does not imply an access edge.
5. Search by memory name and move between the unified primary diagram and supporting context while retaining stable identities.

## 8. Validate Canonical Documentation

When maintained documentation diagram JSON changes, validate and deliver each changed source before the site build:

```bash
node tools/archify/bin/archify.mjs validate architecture docs/<name>.json \
  --quality showcase --repo-root . --json
node tools/archify/bin/archify.mjs deliver architecture docs/<name>.json \
  docsite/static/diagrams/<name>/index.html \
  --quality showcase --repo-root . --json
node tools/archify/bin/archify.mjs visual-check \
  docsite/static/diagrams/<name>/index.html --json
```

Then run:

```bash
npm ci --prefix docsite
npm run typecheck --prefix docsite
npm run build --prefix docsite
```

Expected outcome: canonical docs describe the resulting memory-first experience, changed diagrams pass showcase validation/delivery with zero warnings, and the Docusaurus site builds with all embedded assets. Report browser-based visual checks accurately if skipped.

## 9. Final Regression and Worktree Review

```bash
cargo test
cargo fmt --check
git status --short
```

Expected outcome: the broader Rust suite passes subject to truthfully reported external-validator skips, formatting is clean, and every changed or generated file is expected. Generated visualization output and delivered docsite artifacts remain ignored.
