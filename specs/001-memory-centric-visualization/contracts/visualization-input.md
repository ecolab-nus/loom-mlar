# Contract: Visualization Input Compatibility

## Purpose

Define how the memory-centric planner consumes the existing renderer-neutral input without changing its public contract.

## Accepted Input

- Schema version: exactly `mlar.visualization.v1`.
- Canonical schema: `schemas/mlar-visualization-v1.schema.json`.
- Serialization: human-readable YAML accepted by `mlar-archify build`.
- Compatibility: every document valid before this feature remains valid after it.

## Authoritative Semantics

1. `architecture.root_scope` identifies the root of a connected, acyclic scope hierarchy.
2. A component's `scope` identifies ownership and hierarchy context; it does not imply memory access.
3. A component with `kind: memory` is one canonical memory access endpoint. Its `id` remains stable in every output view.
4. A memory's recursive `region` describes structural array/bank layers. These layers may be rendered with deterministic presentation IDs but are not independent access endpoints.
5. `read` is directional from memory to processor/data mover.
6. `write` is directional from processor/data mover to memory.
7. `requires` and `network_attachment` remain supporting source relationships and must be covered by at least one output view.
8. Dimensions, replication factors, and symbolic expression text are metadata; replicated instances are never expanded.

## Validation and Diagnostics

The adapter must reject an input before planning when:

- it fails the v1 schema;
- a canonical ID is duplicated;
- the root scope is missing or has a parent;
- a scope parent is unknown, cyclic, or disconnected from the root;
- a component references an unknown scope; or
- a relationship endpoint is unknown.

Rust projection errors for missing or ambiguous named memory references remain authoritative and unchanged. The adapter must not repair or infer unresolved relationships.

## Non-Goals

- No new schema fields or version.
- No direct processor/data-mover references to nested array/bank layers.
- No cache, coherence, reachability, or transit semantics inferred from scope containment.
- No renderer/layout information added to the Rust architecture or YAML contract.

## Future Version Trigger

A new visualization schema version is required if nested array/bank layers must become canonical cross-view entities or direct access endpoints. That change also requires an architecture-level reference model and explicit migration guidance; it cannot be implemented as adapter-only inference.

