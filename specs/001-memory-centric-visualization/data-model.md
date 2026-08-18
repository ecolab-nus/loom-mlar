# Phase 1 Data Model: Memory-Centric Visualization

## Model Boundary

The feature does not add a new persistent source model. It derives memory-centered presentation entities from the existing `mlar.visualization.v1` document. Canonical IDs and source relationships remain authoritative; derived entities exist only while planning the generated bundle.

## Source Entities

### Scope

Represents an ownership level in the architecture.

| Field | Meaning | Validation |
| --- | --- | --- |
| `id` | Canonical scope identity | Unique across all document IDs |
| `name` | Human-readable scope name | Non-empty |
| `parent_scope` | Owning scope, absent for root | Must reference a known scope; must be acyclic and connected to root |
| `dimensions` | Symbolic or concrete replication axes | Preserve expression text; do not expand instances |
| `replication_factor` | Concrete instance count when available | Metadata only |

### Canonical Memory Component

Represents a registered memory region and the only valid processor/data-mover access endpoint.

| Field | Meaning | Validation |
| --- | --- | --- |
| `id` | Stable memory identity | Unique canonical ID; reused unchanged across views |
| `scope` | Owning scope | Must reference a known scope |
| `name` | Display name | Non-empty; uniqueness is not assumed |
| `dimensions` | Effective architecture and region dimensions | Preserve symbolic text and constants |
| `region` | Recursive array/bank structure | Must match one valid recursive variant |
| `total_size_bytes` | Concrete aggregate size when known | Null is valid for symbolic sizes |

### Executable Component

Represents either a compute processor or data mover.

| Field | Meaning | Validation |
| --- | --- | --- |
| `id` | Canonical actor identity | Unique canonical ID |
| `kind` | `processor` or `data_mover` | Preserve kind visibly in every access view |
| `scope` | Owning scope | Must reference a known scope |
| `name` | Display name | Non-empty; uniqueness is not assumed |
| `effect` | Data effect | Preserve as descriptive metadata |
| `functions` | Supported function names | Preserve for detail labels |

### Source Relationship

Represents a canonical directional relationship.

| Kind | Source | Target | Meaning |
| --- | --- | --- | --- |
| `read` | Memory | Executable component | Actor reads from exact memory endpoint |
| `write` | Executable component | Memory | Actor writes to exact memory endpoint |
| `requires` | Executable component | Resource | Actor depends on shared resource |
| `network_attachment` | Network | Memory | Network attaches to exact memory endpoint |

Every endpoint must reference a canonical component. Hierarchy never changes or broadens these relationships.

## Derived Planning Entities

### Scope Path

Ordered root-to-owner list of scope IDs and names for a canonical component.

- Derived solely by following `parent_scope`.
- Used for visible context, catalog search, filtering, boundaries, and duplicate-name disambiguation.
- Must terminate at the declared root scope.

### Memory Layer Node

Presentation of one recursive array or bank value inside a canonical memory.

| Field | Meaning | Validation |
| --- | --- | --- |
| `presentation_id` | Stable derived identity | Canonical memory ID plus structural path; never based only on name |
| `canonical_memory_id` | Owning access endpoint | Must reference one canonical memory |
| `parent_layer_id` | Containing array layer | Null only for outermost layer |
| `layer_kind` | Array or bank | Matches source region variant |
| `name` | Optional layer name | May be absent or duplicated |
| `dimensions` | Array axes | Required for arrays, empty for banks |
| `capacity` | Bank capacity expression | Required for banks |
| `block_size` | Bank access granularity | Optional |
| `total_size_bytes` | Concrete layer size | Null allowed for symbolic structure |
| `is_access_endpoint` | Whether access edges may terminate here | True only for the outer canonical memory representation |

Derived layer IDs are stable within the same canonical source structure but are not added to source coverage counts.

### Unified Memory View

The default bounded diagram for architectures whose complete memory-centric
projection fits within 12 primary nodes.

| Field | Meaning |
| --- | --- |
| `id` | Stable root view ID (`memory-hierarchy-1`) |
| `canonical_memory_ids` | Every canonical memory in the architecture |
| `layer_nodes` | Every derived recursive array/bank layer |
| `actor_ids` | Every processor/data mover with direct memory access |
| `scope_boundaries` | Ownership and replication context for memories/layers |
| `contains_connections` | Presentation-only structural containment |
| `source_relationship_ids` | Exact canonical relationship segments rendered as unlabeled directional arrows |
| `legend` | Automatic role legend using `Memory`, `Processor`, and `Data Mover` for the visual types present |

Validation rules:

- Emitted only when the union of canonical memories, derived layers, and
  connected actors contains at most 12 nodes.
- Replaces separate hierarchy, structure, and access diagrams when emitted.
- Each actor appears once with all directly connected canonical memory endpoints.
- Actors with different source and destination hierarchy levels occupy an intervening column, so the visual route is source memory → actor → destination memory.
- Canonical `read`/`write` kinds determine arrow direction internally but are not repeated as edge labels.
- A subtitle defines arrows as source-memory input → actor → destination-memory output and boundaries as architecture scopes.
- Containment never implies access and remains separate from source coverage.

### Memory Hierarchy Overflow Window

A bounded overflow view of ownership and recursive containment, used only when
the unified memory view would exceed 12 nodes.

| Field | Meaning |
| --- | --- |
| `id` | Deterministic view ID from root scope/memory and page index |
| `primary_scope_id` | Scope used for gallery placement |
| `canonical_memory_ids` | Source memories represented in the window |
| `layer_nodes` | Derived structural nodes represented in the window |
| `scope_boundaries` | Ownership context for represented memories |
| `contains_connections` | Presentation-only containment edges |
| `breadcrumb_ids` | Stable repeated parent context when partitioned |

Validation rules:

- At most 12 primary nodes.
- Every canonical memory appears in at least one hierarchy window, including unconnected memories.
- Containment connections never enter the source relationship coverage set.
- No aggregate or replicated-instance nodes are generated.

### Actor Access Unit

One executable component and all of its direct memory access facts, considered atomically for packing.

| Field | Meaning |
| --- | --- |
| `actor_id` | Canonical processor/data-mover ID |
| `anchor_memory_id` | Memory whose page includes this unit |
| `endpoint_memory_ids` | All canonical memories connected to the actor by read/write relationships |
| `relationship_ids` | All canonical read/write relationships among the actor and endpoints |
| `source_memory_ids` | Canonical memories whose relationship arrows enter the actor |
| `destination_memory_ids` | Canonical memories whose relationship arrows leave the actor |

Validation rules:

- The anchor must occur in `endpoint_memory_ids`.
- The actor appears once per view even when both read and write relationships exist.
- All endpoints are exact canonical memories; no ancestor/descendant inference is allowed.
- A data mover with source and destination shows both unlabeled directional segments whenever its unit fits in one view.

### Memory Access Overflow Window

A bounded overflow diagram anchored on one canonical memory and containing one
or more actor access units. It is not generated when the unified view fits.

| Field | Meaning |
| --- | --- |
| `id` | Deterministic anchor/page ID |
| `anchor_memory_id` | Repeated canonical center across pages |
| `actor_units` | Whole access units packed into this page |
| `component_ids` | Unique canonical memories and actors displayed |
| `source_relationship_ids` | Exact directional route segments displayed without access-kind labels |
| `primary_scope_id` | Anchor memory's owning scope |

Validation rules:

- At most 12 unique primary nodes.
- Units are ordered deterministically by actor ID before packing.
- If one unit exceeds the limit, endpoint paging retains both the anchor and actor on every page.
- Across all access windows, every source read/write relationship appears at least once.

### Resource, Network, Or Scope Window

Secondary bounded view stored under the compatibility-stable `supporting_context` section ID for resource dependencies, network attachments, or canonical components/scopes not otherwise covered.

- Uses canonical component and relationship IDs.
- Contains at most 12 primary nodes.
- Never becomes the default gallery view.
- Ensures complete source coverage without crowding memory hierarchy/access views.
- Uses a purpose-specific title: processor/data-mover resources, network attachments, unconnected components in a named architecture scope, or architecture scopes without components.

### Gallery Catalog Entry

Navigation metadata for one delivered Archify diagram.

| Field | Meaning |
| --- | --- |
| `id`, `title`, `html` | Stable view identity and delivered artifact |
| `section` | Unified/overflow `memory_hierarchy`, overflow-only `memory_access`, or compatibility-stable `supporting_context` displayed as `Resources, networks, and scopes` |
| `scope_id`, `scope_path` | Ownership/filter context |
| `memory_ids`, `memory_names` | Search and optional memory-filter context |
| `component_count`, `relationship_count` | Review metadata |

The first root-scope `memory_hierarchy` entry is the default. Catalog metadata supports navigation only; rendered architecture graphics remain inside delivered Archify HTML.

## Relationships

```text
Scope 1 ──owns──> 0..* Canonical Memory Component
Scope 1 ──owns──> 0..* Executable Component
Scope 0..1 ──parents──> 0..* Scope
Canonical Memory Component 1 ──contains──> 1..* Memory Layer Node
Canonical Memory Component * <──read/write──> * Executable Component
Unified Memory View 1 ──presents──> * Canonical Memory Component / Memory Layer Node / Executable Component
Canonical Memory Component 1 ──anchors──> 0..* Memory Access Overflow Window
Memory Hierarchy Overflow Window * ──presents──> * Canonical Memory Component / Memory Layer Node
Memory Access Overflow Window * ──packs──> * Actor Access Unit
Gallery Catalog Entry 1 ──opens──> 1 delivered Archify diagram
```

## Build Lifecycle

The bundle has no mutable user state. Its build lifecycle is:

1. **Loaded**: YAML parsed.
2. **Validated**: v1 schema and canonical references pass.
3. **Projected**: scope paths, memory layers, and access units derived.
4. **Planned**: one unified memory view is created when it fits; otherwise bounded hierarchy/access overflow windows are created, followed by purpose-specific resource, network, and scope views.
5. **Delivered**: each specification passes showcase validation and becomes standalone HTML.
6. **Reported**: manifest and conversion report prove coverage; the build fails if omissions exist.
