# Phase 0 Research: Memory-Centric Visualization

## Decision 1: Preserve `mlar.visualization.v1`

**Decision**: Keep the Rust visualization projection, public re-exports, v1 schema, and tracked normalized YAML shape unchanged for this feature. Implement the memory-centric behavior in the adapter's derived view model.

**Rationale**: The existing document already contains all required source facts: stable top-level memory component IDs, scope ownership and parentage, replication metadata, recursive array/bank details, and directional read/write relationships. Keeping the input contract unchanged preserves existing consumers and avoids migration work that delivers no additional user value.

**Alternatives considered**:

- Add optional hierarchy fields to v1: rejected because the strict schema is a compatibility boundary and older validators would reject newly produced fields.
- Introduce `mlar.visualization.v2`: reserved for a future requirement to make nested array/bank layers independently addressable access endpoints. That semantic capability does not exist in the current architecture model.
- Maintain a separate hand-authored memory hierarchy: rejected because it could drift from the architecture model and violates the single-source and simplicity principles.

## Decision 2: Distinguish Canonical Memory Endpoints from Structural Layers

**Decision**: A top-level memory component remains the only canonical access endpoint. Recursive array/bank values become presentation-only structural layer nodes with deterministic IDs derived from the canonical memory ID and structural path.

**Rationale**: Processor references name one registered memory region and the exporter resolves read/write relationships to that component. Nested array/bank values have no canonical source IDs and are not independently addressable. Derived IDs make their structure stable across views without inventing access semantics.

**Alternatives considered**:

- Render only one memory node and omit recursive details: rejected because users must see dimensions, capacity, block size, size, and nested structure.
- Treat every nested layer as an access endpoint: rejected because it would misrepresent the current architecture contract.
- Use display names to derive IDs: rejected because names may be absent or duplicated; structural paths under the canonical ID are deterministic and collision-resistant.

## Decision 3: Represent Hierarchy as Containment, Not Access

**Decision**: Derive hierarchy presentation from scope ancestry and recursive region containment. Use scope boundaries and derived `contains` connections; never generate read/write edges from containment.

**Rationale**: A memory in an ancestor scope is not necessarily the parent, cache backing store, or transit path for every memory in a descendant scope. Scope ancestry provides ownership context, while only explicit source relationships establish access.

**Alternatives considered**:

- Infer DRAM-to-L1 edges from scope nesting: rejected because multiple memories may share a scope and hierarchy alone does not define access.
- Always draw one global memory graph regardless of size: rejected because realistic models can exceed the 12-node readability boundary. A unified graph is preferred when it fits.
- Retain only scope overview pages: rejected because scope nodes would remain the primary organizing elements and unconnected memory hierarchy would be hard to inspect.

## Decision 4: Unify Hierarchy, Structure, and Access When They Fit

**Decision**: First project canonical memories, recursive layers, and all directly connected actors into one diagram. Emit that single primary diagram whenever the union has at most 12 nodes. Group all read/write relationships by actor so each actor appears once with all of its canonical memory endpoints and original directional edges. Use memory-anchored access pages only as overflow for larger models.

**Rationale**: Separating hierarchy, recursive structure, and access forced users to move among diagrams even when the representative model has only nine relevant nodes. One diagram is the smaller mechanism and makes DRAM, L1, banks, processors, movers, and routes directly comparable. The overflow path remains justified only by the hard readability bound.

**Alternatives considered**:

- Merely reorder the current read/write pages: rejected because the connection remains fragmented.
- One page per relationship: rejected because it prevents route tracing and creates excessive pages.
- Always create separate hierarchy, structure, and access pages: rejected because it fragments a model that fits cleanly in one diagram.
- One page per entire architecture without a size check: rejected because it can violate the node cap and degrade readability.

## Decision 5: Partition Deterministically Without Collapsing Entities

**Decision**: Attempt the unified memory view first. Only if it exceeds 12 nodes, split hierarchy subtrees and access actor units into deterministic chunks. Repeat canonical anchors and breadcrumb context across chunks; never replace entities with aggregate nodes or expand replicated instances.

**Rationale**: Repetition with stable IDs preserves identity and navigation while satisfying the constitution's hard readability and replication rules. Deterministic ordering makes output reproducible and testable.

**Alternatives considered**:

- Expand replicated mesh instances: rejected because output size would scale with instance count and violate project policy.
- Collapse many actors into summary nodes: rejected because distinct source entities would become untraceable.
- Allow oversized views: rejected because the 12-node limit is a compatibility and quality gate.

## Decision 6: Keep Supporting Context Secondary but Complete

**Decision**: Preserve resource, network, and otherwise uncovered components in secondary supporting-context views, chunked by unique nodes. Keep source coverage accounting distinct from presentation-only containment edges.

**Rationale**: The feature is memory-centric, not memory-exclusive. Network and resource relationships explain movement constraints and every source component/relationship must remain discoverable.

**Alternatives considered**:

- Remove non-memory views: rejected because it would lose modeled information and fail completeness checks.
- Show all supporting context on every memory page: rejected because it would crowd the primary views and frequently exceed the node cap.

## Decision 7: Retain the Static Gallery and Archify Renderer Boundary

**Decision**: Make the unified hierarchy-and-access diagram the gallery default, list overflow access only when required, enrich catalog search/filter metadata, and retain hashes, keyboard navigation, scope filtering, standalone opening, and iframe embedding of delivered Archify HTML.

**Rationale**: The existing shell already provides the needed navigation without drawing architecture graphics. Enhancing its catalog is sufficient for linked bounded views and avoids a second renderer.

**Alternatives considered**:

- Build an interactive custom graph in the gallery: rejected because it duplicates rendering behavior and violates the renderer boundary.
- Require a backend service: rejected because all source data and views are static and the current deployable bundle already meets the use case.

## Decision 8: Validate with Layered Evidence

**Decision**: Use focused adapter tests first, existing Rust visualization regressions second, then a complete 2D mesh bundle build with coverage/receipt inspection, optional browser visual checking, and canonical documentation checks.

**Rationale**: The behavioral change is concentrated in adapter planning and gallery organization, while the unchanged Rust/schema path still needs regression evidence. The representative sample exercises DRAM, replicated L1, compute processors, data movers, resources, and network context.

**Alternatives considered**:

- Run only unit tests: rejected because they do not prove showcase delivery, source coverage, or bundle navigation.
- Run the full repository suite before focused checks: rejected because focused failures are faster to diagnose; broader validation remains a final gate.
