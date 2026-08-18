# Contract: Memory-Centric Visualization Bundle

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

1. **Memory hierarchy and access** (`memory_hierarchy`): ownership, recursive memory structure, connected processors/data movers, and exact directional access together. The first root-scope view is the default.
2. **Additional memory access** (`memory_access`): overflow pages created only when a complete unified view would exceed 12 primary nodes.
3. **Supporting context** (`supporting_context`): resources, networks, and canonical entities not covered by the primary view.

All authored section labels, titles, diagnostics, and navigation strings are English.

## Unified Memory Hierarchy And Access View

- When all canonical memories, recursive layers, and directly connected actors total at most 12 primary nodes, the planner emits exactly one primary view and no separate structure or access views.
- Every canonical memory appears in the primary view, even when unconnected.
- Scope boundaries and paths show ownership hierarchy.
- Recursive array/bank structure is shown with deterministic derived layer IDs and `contains` connections.
- Every directly connected processor and data mover appears in the same view, once per canonical actor identity.
- Original relationship direction is preserved as unlabeled arrows; the presentation does not add `read` or `write` edge text.
- Each connected actor is placed between its source and destination hierarchy columns whenever the endpoints differ, forming source memory → actor → destination memory.
- A data mover includes all its canonical memory endpoints, allowing complete source → mover → destination routes to appear in the unified view.
- Derived containment is not reported as a canonical source relationship and never implies access.
- Symbolic values show their expression text; unavailable concrete totals remain explicitly unknown rather than fabricated.
- No edge is inferred from hierarchy, same-name matching, resource dependencies, or network attachments.

## Overflow Views

- Separate hierarchy windows and memory-anchored access pages are created only when the unified view would contain more than 12 primary nodes.
- Wide/deep structures split deterministically with stable breadcrumb context.
- Access overflow pages retain whole actor units, all direct canonical memory endpoints, and exact unlabeled directional routes.
- Overflow never expands replicated instances or replaces distinct canonical entities with aggregate nodes.

## Supporting Context Views

- Every source `requires` and `network_attachment` relationship appears at least once.
- Networks, resources, and actors without memory access remain discoverable.
- Supporting views are secondary and are not chosen as the default.

## Gallery Behavior

- Default view: first root-scope unified memory hierarchy-and-access view.
- Search matches view title, section, scope path, canonical memory name, and memory identity.
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
