# Feature Specification: Memory-Centric Visualization

**Feature Branch**: `dev`

**Created**: 2026-08-17

**Status**: Draft

**Input**: User description: "Refactor the visualization to be memory-centric: emphasize memory regions, the processors and data movers connected to them at each level, and the hierarchy of memory regions."

## Clarifications

### Session 2026-08-18

- Q: Which components receive dedicated focused views, and what must each view contain? → A: Every memory, processor, and data mover receives a component view containing its directly connected canonical components and required resources; resources and networks appear as neighbors rather than receiving dedicated views by default, while otherwise uncovered components are grouped by owning scope.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Understand the Memory Hierarchy (Priority: P1)

As a hardware architect, I want the visualization to lead with memory regions and their hierarchy so that I can understand the architecture's storage levels before examining the components that use them.

**Why this priority**: The memory hierarchy is the organizing structure for every other part of this feature. Without it, the visualization is not meaningfully memory-centric.

**Independent Test**: Open a visualization of an architecture containing memory regions at multiple nested levels and verify that the default `System View` shows every region, its recursive layers, its hierarchy context, and its connected processors/data movers whenever those entities fit within the 12-node limit.

**Acceptance Scenarios**:

1. **Given** an architecture with system-level DRAM and a replicated child scope containing L1 memory, **When** a user opens the visualization, **Then** DRAM and L1 are presented as memory regions in a clear hierarchy and L1's replication is shown as metadata rather than expanded instances.
2. **Given** a memory region composed of nested region structures, **When** the user examines that region, **Then** the nested structure and each level's available name, dimensions, capacity, and size information are visible.
3. **Given** an unconnected memory region, **When** the visualization is generated, **Then** the region still appears in its correct hierarchy and is clearly distinguishable from connected regions.
4. **Given** the representative DRAM/L1 architecture fits within the readability limit, **When** the visualization is generated, **Then** `System View` contains DRAM, L1, their bank layers, and all directly connected processors/data movers rather than separate hierarchy, structure, and access diagrams.

---

### User Story 2 - See Who Accesses Each Memory Level (Priority: P1)

As a hardware architect, I want each memory region to show the processors and data movers connected to that exact level so that I can verify access boundaries and data movement responsibilities.

**Why this priority**: Connectivity is the central question the memory-centric view must answer: who can read from or write to each memory region.

**Independent Test**: Use a model where compute processors access L1 and data movers connect L1 with DRAM; for each memory region, compare the displayed neighbors and directions with the declared connections in the source architecture.

**Acceptance Scenarios**:

1. **Given** a compute processor whose source and destination are both L1, **When** the user examines L1, **Then** the processor appears once between the memory hierarchy levels and the arrows form L1 → processor → L1 without `read` or `write` text labels.
2. **Given** a data mover from DRAM to L1, **When** the user examines either endpoint, **Then** the same data mover is placed between DRAM and L1 and the arrows form DRAM → data mover → L1.
3. **Given** processors connected to different levels of a memory hierarchy, **When** the hierarchy is viewed, **Then** each processor is associated only with the exact region or regions it accesses and is not implied to access ancestors or descendants.
4. **Given** multiple connections between the same component and memory region, **When** they represent distinct access meanings, **Then** each meaning remains discoverable without duplicating the component's identity.
5. **Given** a memory, processor, or data mover, **When** the user opens its component view, **Then** the selected component is the stable anchor and every canonical component joined to it by a direct source relationship is shown once.
6. **Given** a processor or data mover with resource requirements, **When** its component view opens, **Then** all directly required resources appear alongside its source and destination memories.

---

### User Story 3 - Trace Movement Across Memory Levels (Priority: P2)

As a performance engineer, I want to follow a data-movement route between memory levels so that I can quickly understand how data reaches compute and where network or shared-resource constraints apply.

**Why this priority**: Cross-level tracing turns the hierarchy and direct connections into an actionable system-level understanding, while remaining independently useful after the core memory and connectivity views exist.

**Independent Test**: Starting from either endpoint in a multi-level sample, trace a declared DRAM-to-L1 movement through its data mover and confirm that any related network and resource context remains discoverable.

**Acceptance Scenarios**:

1. **Given** a data mover connecting two memory regions at different hierarchy levels, **When** the user follows the connection from one endpoint, **Then** the other endpoint, movement direction, and mover identity can be reached without searching unrelated component views.
2. **Given** a connected processor or data mover with resource relationships, **When** the user opens its component view, **Then** its memory inputs, memory outputs, and directly required resources appear together without adding transitive neighbors.
3. **Given** a large architecture that cannot fit in one readable view, **When** the visualization is organized into multiple views, **Then** users can navigate among memory-centered views without losing the identity or hierarchy context of repeated entities.

### Edge Cases

- A memory region has no processor or data-mover connections.
- A processor reads and writes the same memory region.
- A data mover has the same memory region as both source and destination.
- A component connects to multiple memory regions at the same level or across different levels.
- Different scopes contain memory regions with the same display name.
- A hierarchy contains symbolic dimensions or sizes that cannot be reduced to concrete values.
- A hierarchy is deep or wide enough to exceed the readability limit of a single view.
- A component references a memory region that cannot be resolved uniquely; visualization generation must report the problem rather than show a misleading connection.
- A resource or network has no relationship to a memory, processor, or data mover; it must remain discoverable in a fallback view named for its owning scope.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The visualization MUST use `System View` as the default entry point. It MUST combine memory hierarchy, recursive memory structure, connected processors/data movers, access routes, and architecture scope boundaries whenever those entities fit within the 12-node limit.
- **FR-002**: The visualization MUST show the parent/child hierarchy of memory regions using the architecture's ownership hierarchy and each region's nested structure.
- **FR-003**: Every distinct memory region MUST retain a stable identity even when it appears in more than one view, has the same display name as another region, or is accessed by multiple components.
- **FR-004**: Each memory region MUST expose its available name, hierarchy context, dimensions, replication factor, capacity, block size, and total size without requiring replication to be expanded into individual instances.
- **FR-005**: Every canonical memory, processor, and data mover MUST have a component view anchored on that component.
- **FR-006**: The visualization MUST distinguish compute processors from data movers wherever they are shown, and the primary diagram legend MUST name the three MLAR roles `Memory`, `Processor`, and `Data Mover` rather than Archify's generic web-system type names.
- **FR-007**: Each processor or data mover MUST be presented as a route from its exact source memory through the actor to its exact destination memory; arrow direction alone MUST convey the access direction, without `read` or `write` edge labels.
- **FR-008**: A data mover that connects two memory regions MUST show both endpoints and the complete source memory → data mover → destination memory route.
- **FR-009**: The visualization MUST NOT infer access from hierarchy alone; a connection to one memory region MUST NOT imply a connection to its ancestors, descendants, or siblings.
- **FR-010**: Users MUST be able to follow directional arrows from a source memory to a connected component and from that component to its destination memory while preserving entity identity.
- **FR-011**: Unconnected memory regions MUST remain visible in their correct hierarchy and be recognizable as having no direct processor or data-mover connections.
- **FR-012**: The visualization MUST preserve every modeled component and relationship in at least one view across `System View`, `Component Views`, and scope-owned fallback pages for otherwise uncovered components.
- **FR-013**: No individual semantic view MUST contain more than 12 primary nodes; larger hierarchies or connection sets MUST be split into navigable, memory-centered views.
- **FR-014**: Replicated scopes and memory arrays MUST be represented with dimensions and instance-count metadata rather than by expanding every instance.
- **FR-015**: Ambiguous or missing memory references MUST prevent the affected connection from being presented as valid and MUST produce a diagnostic that identifies the unresolved component and reference.
- **FR-016**: Previously valid visualization source documents MUST either remain usable or receive an explicit compatibility version and migration guidance.
- **FR-017**: The canonical visualization content MUST remain available as a human-readable, versioned textual artifact that can be inspected independently of the rendered views.
- **FR-018**: All project-authored labels, diagnostics, and navigation for the memory-centric experience MUST be in English.
- **FR-019**: The planner MUST NOT create separate system-level hierarchy, recursive-structure, or memory-access diagrams when their union contains at most 12 primary nodes; partitioned `System View` overflow pages are permitted only when the combined system view would exceed that limit.
- **FR-020**: In a unified view, each processor or data mover whose source and destination are at different memory hierarchy levels MUST be positioned spatially between those canonical memory regions.
- **FR-021**: The unified view MUST explicitly explain that arrows show source-memory input through a processor or data mover to destination-memory output, while boundaries show architecture scope ownership and do not imply access.
- **FR-022**: The gallery MUST expose two primary reader-facing sections: `System View` followed by `Component Views`. It MUST NOT expose `Memory hierarchy and access`, `Supporting context`, or `Resources, networks, and scopes` as section names.
- **FR-023**: A component view MUST contain its anchor plus every canonical component connected directly to the anchor by `read`, `write`, `requires`, or `network_attachment`. It MUST NOT add transitive neighbors or infer relationships from scope or hierarchy.
- **FR-024**: Processor and data-mover component views MUST show source-memory → actor → destination-memory direction without `read` or `write` labels and MUST also show every directly required resource with its explicit `requires` relationship.
- **FR-025**: Memory component views MUST show every directly connected processor/data mover and any directly attached network; resource and network components MUST NOT receive dedicated component views by default.
- **FR-026**: If a component view would exceed 12 primary nodes, it MUST split deterministically while repeating the canonical anchor and preserving every direct relationship across the resulting pages.
- **FR-027**: Components not covered by `System View` or any component view MUST be grouped in a fallback component view titled with their owning architecture scope.
- **FR-028**: Scope boundaries in a `System View` MUST preserve the architecture hierarchy without ambiguous partial overlap: an ancestor boundary MUST contain every displayed descendant-scope component, and displayed sibling-scope regions MUST occupy disjoint layout bands.

### Key Entities *(include if feature involves data)*

- **Memory Region**: A named or structural storage area with hierarchy context, dimensions, replication, and size characteristics; it can be a parent, child, or endpoint of access relationships.
- **Memory Hierarchy Relationship**: A parent/child relationship that locates one memory region or nested region structure relative to another without implying access.
- **Compute Processor**: An executable architecture component that directly reads from and/or writes to one or more memory regions.
- **Data Mover**: An executable architecture component with a source memory region and destination memory region, representing directed movement within or across hierarchy levels.
- **Memory Access Connection**: One segment of an unlabeled directional route from source memory through a compute processor or data mover to destination memory.
- **System View**: The default bounded presentation containing canonical memories, recursive structural layers, connected processors/data movers positioned between hierarchy levels, containment, exact source-to-destination routes, and hierarchically nested, non-partially-overlapping architecture scope boundaries together.
- **Component View**: A bounded one-hop neighborhood anchored on one canonical memory, processor, or data mover and containing its direct source relationships, endpoint components, required resources, and ownership scope.
- **Scope Fallback View**: A bounded component-view page that preserves otherwise uncovered canonical components and names their owning architecture scope.
- **Semantic View**: A bounded primary or overflow presentation preserving canonical identities when the complete unified memory view cannot fit.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In the representative multi-level architecture sample, 100% of modeled memory regions appear in the visualization with the correct parent/child placement and replication context.
- **SC-002**: In reference comparisons, 100% of declared processor and data-mover routes appear as source memory → actor → destination memory with the correct arrow direction, no `read`/`write` text labels, and zero inferred or duplicate connections.
- **SC-003**: At least 90% of evaluators can identify which processors access a selected memory level and which data movers connect it to other levels within 60 seconds, without consulting the source architecture.
- **SC-004**: At least 90% of evaluators can trace a declared cross-level data-movement route from source memory to destination memory correctly on their first attempt.
- **SC-005**: Every semantic view contains at most 12 primary nodes, and replicated structures remain understandable without displaying individual instances.
- **SC-006**: Automated completeness checks confirm that 100% of modeled components and relationships remain discoverable across the generated views.
- **SC-007**: No previously valid reference visualization becomes unusable without either continued compatibility or documented migration guidance.
- **SC-008**: The representative 2D mesh sample produces exactly one `System View` containing DRAM, L1, both bank layers, and every directly connected processor/data mover positioned between DRAM and L1, including DRAM → `dram_l1_noc0` → L1 and L1 → `l1_dram_noc1` → DRAM.
- **SC-009**: Every generated system diagram uses the visible legend labels `Memory`, `Processor`, and `Data Mover`; the gallery defaults to `System View` and groups all focused one-hop diagrams under `Component Views`.
- **SC-010**: For every memory, processor, and data mover in the representative sample, automated comparison confirms that its component views collectively include 100% of its direct source relationships and zero transitive or inferred relationships.
- **SC-011**: Automated scope-layout checks confirm that every displayed child-scope boundary is contained by each displayed ancestor boundary and that sibling-scope boundary row ranges have at least one clear grid row between them.

## Assumptions

- "Memory-centric" changes how architecture information is organized and explored; it does not change the meaning of the underlying architecture, schedule, performance, or compiler-facing models.
- Memory hierarchy is derived from the existing scope ownership hierarchy and the recursive structure of each memory region. This feature does not introduce a separate, manually maintained hierarchy that could disagree with the architecture model.
- Only explicitly modeled source and destination regions establish access. The visualization does not infer cache coherence, transitive reachability, or implicit access through a parent memory level.
- Existing directional `read` and `write` source relationships remain canonical interchange facts, but the Archify presentation uses them only to form unlabeled source memory → actor → destination memory arrows.
- Compute processors and data movers are both important neighbors of memory, but remain visibly different component roles.
- Network and shared-resource information remains in scope as direct neighbors inside component views because it can explain data-movement constraints and is required for a complete representation.
- The representative 2D mesh architecture is the primary acceptance fixture for multi-level memory, compute access, data movement, replication, resources, and networks.
- Interactive selection is not required. One combined static system diagram is preferred; bounded linked overflow and component views are used when inspecting one component or when the 12-node limit makes one diagram impossible.
