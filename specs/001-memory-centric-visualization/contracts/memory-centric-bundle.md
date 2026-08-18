# Contract: Memory-Centric Visualization Bundle

> **Historical record.** This feature is implemented and closed. Canonical documentation lives in `docs/` and `README.md`; this directory is retained for decision traceability only.

## Purpose

Define observable behavior of the static bundle generated from a valid `mlar.visualization.v1` document.

## Bundle Contents

The output remains a deployable static directory containing:

- a gallery `index.html`;
- editable Archify JSON specifications;
- delivered standalone Archify HTML diagrams;
- `bundle-manifest.json` with validation, delivery, and optional visual-check receipts; and
- `conversion-report.json` with source coverage and partition policy.

## View Sections and Order

1. **System View** (`system_view`): ownership, recursive memory structure, connected processors/data movers, and exact directional access together. The first root-scope view is the default.
2. **Component Views** (`component_views`): one exact one-hop view for every memory, processor, and data mover, plus owning-scope fallbacks for otherwise uncovered entities.

All authored section labels, titles, diagnostics, and navigation strings are English.

## System View

- When all canonical memories, recursive layers, and directly connected actors total at most 12 primary nodes, the planner emits exactly one primary view and no separate structure or access views.
- Every canonical memory appears in the primary view, even when unconnected.
- Scope boundaries and paths show ownership hierarchy.
- Recursive array/bank structure is shown with deterministic derived layer IDs and `contains` connections.
- Every directly connected processor and data mover appears in the same view, once per canonical actor identity.
- The automatic role legend names present visual types `Memory`, `Processor`, and `Data Mover`, not `Database`, `Backend`, or `Message bus`.
- The subtitle explains that arrows show source-memory input through an actor to destination-memory output and that boundaries show architecture scope ownership.
- Original relationship direction is preserved as unlabeled arrows; the presentation does not add `read` or `write` edge text.
- Each connected actor is placed between its source and destination hierarchy columns whenever the endpoints differ, forming source memory → actor → destination memory.
- A data mover includes all its canonical memory endpoints, allowing complete source → mover → destination routes to appear in the unified view.
- Derived containment is not reported as a canonical source relationship and never implies access.
- Symbolic values show their expression text; unavailable concrete totals remain explicitly unknown rather than fabricated.
- No edge is inferred from hierarchy, same-name matching, resource dependencies, or network attachments.

## System View Overflow

- Separate bounded System View windows are created only when the unified view would contain more than 12 primary nodes. Exact routes remain available in Component Views.
- Wide/deep structures split deterministically with stable breadcrumb context.
- Overflow never expands replicated instances or replaces distinct canonical entities with aggregate nodes.

## Component Views

- Every memory, processor, and data mover has a focused view, including unconnected anchors.
- A focused view contains the anchor, every directly connected canonical component, and all source relationships between them.
- No transitive neighbor or inferred hierarchy/name/scope relationship is added.
- A processor or data mover shows exact memory input/output and every directly required resource.
- A memory shows exact processors/data movers and direct network attachments.
- Resources and networks do not receive dedicated focus views by default.
- More than 11 unique neighbors are partitioned deterministically, with the anchor repeated and each neighbor's relationships kept together.
- Otherwise uncovered components and empty scopes remain discoverable in views explicitly titled for their owning `Architecture Scope`.

## Gallery Behavior

- Default view: first root-scope `System View`.
- Search matches view title, section, scope path, canonical memory metadata, and focused component ID/name/kind.
- Scope filtering remains available; an optional memory filter may be added from catalog metadata.
- URL hashes preserve selected views.
- Previous/next controls and keyboard navigation remain available.
- Each diagram can open independently.
- The shell embeds delivered Archify HTML and contains no architecture SVG or alternative graph renderer.

## Coverage and Bounds

Generation succeeds only when:

- every source scope ID is included in at least one view or ownership boundary tracked by the report;
- every source component ID is included in at least one view;
- every source relationship ID is included in at least one view;
- every diagram contains at most 12 primary nodes;
- no replicated instances are expanded; and
- no distinct canonical entities are replaced by synthetic aggregates.

Presentation-only region nodes and containment connections are tracked separately from canonical source coverage.

## Determinism

Given identical input and tool versions, view IDs, derived layer IDs, section order, chunk membership, component ordering, and coverage output must be identical.
