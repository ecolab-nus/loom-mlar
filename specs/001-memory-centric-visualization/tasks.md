---

description: "Dependency-ordered implementation tasks for memory-centric visualization"
---

# Tasks: Memory-Centric Visualization

**Input**: Design documents from `specs/001-memory-centric-visualization/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Required by the feature's acceptance scenarios and the project constitution. Story-specific tests are written before their implementation tasks.

**Scope guard**: Implement this feature in `tools/mlar-archify/`, its tests, and canonical documentation. Treat `src/visualization/document.rs`, `src/lib.rs`, `schemas/mlar-visualization-v1.schema.json`, and `tests/2d_mesh/2d_mesh_torus.visualization.yaml` as unchanged regression boundaries unless implementation proves the approved v1 contract insufficient and the user approves a scope change.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel after the preceding blocking task because it edits a different file and uses an already-defined contract.
- **[Story]**: Maps the task to User Story 1, 2, or 3 from `spec.md`.
- Every task names the exact file or artifact it affects.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish a reproducible baseline without changing project structure or dependencies.

- [X] T001 Install pinned adapter dependencies with `npm ci --prefix tools/mlar-archify` and verify the Node.js 20 requirement and unchanged dependency set in `tools/mlar-archify/package.json` and `tools/mlar-archify/package-lock.json`
- [X] T002 Run the baseline adapter tests and representative Rust export checks against `tools/mlar-archify/test/planner.test.mjs`, `src/visualization/document.rs`, and `tests/2d_mesh/2d_mesh_torus.visualization.yaml`, recording any pre-existing failure in `specs/001-memory-centric-visualization/tasks.md` before edits

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Add the shared adapter-side projection and coverage mechanisms required by every user story while preserving the v1 input contract.

**⚠️ CRITICAL**: No user-story implementation begins until this phase is complete.

- [X] T003 Implement reusable scope-path, canonical-memory, recursive memory-layer, and actor-access indexes with deterministic derived layer IDs in `tools/mlar-archify/bin/mlar-archify.mjs`
- [X] T004 Extend diagram construction to support scope boundaries, presentation-only containment connections, explicit component positioning, and separate canonical-versus-derived identity metadata in `tools/mlar-archify/bin/mlar-archify.mjs`
- [X] T005 Update manifest and conversion-report accounting so source scopes, components, and relationships remain complete while derived memory layers and containment edges are tracked separately in `tools/mlar-archify/bin/mlar-archify.mjs`

**Checkpoint**: The adapter can derive memory structure and access units from any valid `mlar.visualization.v1` document without changing the Rust exporter or schema.

---

## Phase 3: User Story 1 - Understand the Memory Hierarchy (Priority: P1) 🎯 MVP

**Goal**: Make one combined memory hierarchy-and-access diagram the default experience when it fits, showing scope ownership, recursive array/bank structure, replication metadata, symbolic values, unconnected memories, and connected actors together.

**Independent Test**: Generate a model with root DRAM, replicated child-scope L1, nested array/bank structure, symbolic size metadata, duplicate memory names in separate scopes, and an unconnected memory; verify every memory has stable hierarchy placement, no replicated instances are expanded, and every view has at most 12 primary nodes.

### Tests for User Story 1

- [X] T006 [US1] Add failing hierarchy and gallery tests for root-first defaults, scope boundaries, recursive layers, stable derived IDs, symbolic metadata, duplicate names, unconnected status, and 12-node partitioning in `tools/mlar-archify/test/planner.test.mjs`

### Implementation for User Story 1

- [X] T007 [P] [US1] Implement deterministic `memory_hierarchy` view planning, recursive layer rendering, containment edges, ownership boundaries, unconnected-state labels, and bounded subtree windows in `tools/mlar-archify/bin/mlar-archify.mjs` (superseded by the unified fit-first planner in T028)
- [X] T008 [P] [US1] Make the root memory hierarchy-and-access diagram the default, add section ordering, and index memory names, IDs, and scope paths for search/filter navigation in `tools/mlar-archify/lib/gallery.mjs`
- [X] T009 [US1] Run the focused US1 tests and inspect generated hierarchy specifications from `tools/mlar-archify/test/planner.test.mjs` and `visualization-output/2d-mesh/specifications/` for stable IDs, visible metadata, zero inferred access edges, and the 12-node maximum

**Checkpoint**: User Story 1 works as the hierarchy foundation of the unified view, including architectures with no processor or data-mover connections.

---

## Phase 4: User Story 2 - See Who Accesses Each Memory Level (Priority: P1)

**Goal**: Give each canonical memory a bounded view of every directly connected processor and data mover, preserving exact read/write direction and displaying each actor once.

**Independent Test**: Use a model where processors read and write L1, a data mover connects DRAM to L1, another mover has the same source and destination, and same-named memories exist in different scopes; compare all displayed endpoints and directions with the source relationships and confirm hierarchy alone creates no access.

### Tests for User Story 2

- [X] T010 [US2] Add failing access tests for combined read/write actors, compute-versus-data-mover styling, complete direct endpoint sets, same-memory movers, cross-scope routes, duplicate memory names, no hierarchy-inferred edges, and over-12-neighbor partitioning in `tools/mlar-archify/test/planner.test.mjs`

### Implementation for User Story 2

- [X] T011 [P] [US2] Implement memory-anchored actor-unit grouping, exact directional edge selection, all-endpoint expansion, actor deduplication, and deterministic bounded packing for overflow-only `memory_access` views in `tools/mlar-archify/bin/mlar-archify.mjs`
- [X] T012 [P] [US2] Add memory-access catalog metadata, section labels, memory-aware search results, and linked navigation between repeated canonical identities in `tools/mlar-archify/lib/gallery.mjs`
- [X] T013 [US2] Run the focused US2 tests in `tools/mlar-archify/test/planner.test.mjs` and verify a generated L1 access view contains each compute processor once with both read/write edges and each data mover with only its declared canonical memory endpoints

**Checkpoint**: User Story 2 independently answers which processors and data movers access any selected memory and in which direction.

---

## Phase 5: User Story 3 - Trace Movement Across Memory Levels (Priority: P2)

**Goal**: Preserve complete cross-level movement routes and make associated resources, networks, and otherwise uncovered entities discoverable as secondary context.

**Independent Test**: Starting from either DRAM or L1 in the 2D mesh fixture, trace the declared data mover to the opposite endpoint, open its resource/network context, and verify all canonical source IDs remain covered across bounded views.

### Tests for User Story 3

- [X] T014 [US3] Add failing route, supporting-context, completeness, deterministic-output, English-only, and no-gallery-SVG tests covering resources, networks, orphan components, and large models in `tools/mlar-archify/test/planner.test.mjs`

### Implementation for User Story 3

- [X] T015 [P] [US3] Implement bounded `supporting_context` planning for resource dependencies, network attachments, and uncovered canonical components while preserving complete source-ID coverage in `tools/mlar-archify/bin/mlar-archify.mjs`
- [X] T016 [P] [US3] Add supporting-context ordering and labels while preserving hashes, keyboard controls, standalone opening, iframe embedding, and renderer separation in `tools/mlar-archify/lib/gallery.mjs`
- [X] T017 [US3] Build the full fixture from `tests/2d_mesh/2d_mesh_torus.visualization.yaml` into `visualization-output/2d-mesh/`, then verify DRAM-to-L1 routes, supporting views, showcase receipts, deterministic IDs, zero omissions in `conversion-report.json`, and at most 12 primary nodes per generated specification

**Checkpoint**: All three user stories work independently, and the complete 2D mesh experience is memory-first without losing non-memory source semantics.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Consolidate durable behavior into canonical documentation and complete proportionate regression, visual, and worktree validation.

- [X] T018 [P] Update the memory-first overview, commands, bundle behavior, and renderer-boundary guidance in `README.md` and `tools/mlar-archify/README.md`
- [X] T019 [P] Document canonical endpoint semantics, recursive layer presentation, hierarchy-versus-access rules, and user workflows in `docs/architecture-concepts.md` and `docs/usage.md`
- [X] T020 [P] Update installation, artifact flow, and module-responsibility descriptions for the memory-centric planner in `docs/installation.md`, `docs/project-overview.md`, and `docs/software-architecture.md`
- [X] T021 Review `docs/mlar-project-architecture.json` and `docs/project-overview.json` for inaccurate planner descriptions; update only inaccurate JSON sources, then validate, deliver, and visually check each changed source into `docsite/static/diagrams/` with the vendored `tools/archify/bin/archify.mjs`
- [X] T022 Run documentation typechecking and the complete site build from `docsite/package.json`, confirming all changed Archify diagrams and referenced assets are present in `docsite/build/`
- [X] T023 Run Rust visualization and full regression checks against `src/visualization/document.rs`, `src/lib.rs`, `schemas/mlar-visualization-v1.schema.json`, and `tests/2d_mesh/2d_mesh_torus.visualization.yaml`; confirm those compatibility-boundary files have no unintended semantic diff and finish with `cargo fmt --check`
- [X] T024 Run the complete adapter suite and rebuild `visualization-output/2d-mesh/` with `--visual-check`, verifying all validation/delivery receipts in `bundle-manifest.json` and reporting an unavailable browser as skipped rather than passed
- [X] T025 Execute the timed hierarchy, memory-access, and route-tracing acceptance journeys from `specs/001-memory-centric-visualization/quickstart.md` and record the measured outcomes against SC-003 and SC-004 in `specs/001-memory-centric-visualization/tasks.md`
- [X] T026 Inspect `git status --short` and all task-related diffs, distinguish expected generated or pre-existing changes, and record in `specs/001-memory-centric-visualization/plan.md` whether the completed feature directory is retained as historical or archived after canonical documentation consolidation

---

## Phase 7: Unified Primary Diagram Follow-up

**Purpose**: Apply the follow-up requirement to combine hierarchy, recursive layers, and connected actors whenever the complete memory-centric projection fits the readability limit.

- [X] T027 Add focused tests requiring one primary diagram for the representative-sized model, including canonical memories, recursive bank layers, processors/data movers, containment, and exact read/write edges in `tools/mlar-archify/test/planner.test.mjs`
- [X] T028 Implement fit-first unified memory planning with deterministic hierarchy-level positioning and retain the existing hierarchy/access planners only as over-12-node overflow in `tools/mlar-archify/bin/mlar-archify.mjs`
- [X] T029 Update `spec.md`, `plan.md`, `research.md`, `data-model.md`, bundle contract, quickstart, adapter documentation, and canonical project documentation to make the unified view normative
- [X] T030 Rebuild and validate the 2D mesh bundle, verify exactly one 9-node primary memory diagram, zero omissions, showcase receipts, compatibility-boundary stability, and complete regression checks

---

## Phase 8: Actor Route Layout Follow-up

**Purpose**: Make processors and data movers explicit route nodes between their source and destination memory regions, using arrow direction instead of access-kind labels.

- [X] T031 Add focused forward/reverse route tests requiring actors between DRAM and L1, exact source → actor → destination direction, and no `read`/`write` edge labels in `tools/mlar-archify/test/planner.test.mjs`
- [X] T032 Implement hierarchy-depth memory columns, intervening actor placement, and unlabeled directional access connections in `tools/mlar-archify/bin/mlar-archify.mjs`
- [X] T033 Update feature artifacts and canonical user documentation to specify actor-between-memory routing and arrow-only access semantics
- [X] T034 Rebuild and validate the 2D mesh bundle, inspect both opposite mover routes and node positions, rerun regression checks, and record the evidence below

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: Starts immediately; T002 depends on T001.
- **Foundational (Phase 2)**: Depends on Setup and blocks all stories; T003 → T004 → T005.
- **User Story 1 (Phase 3)**: Starts after T005; T006 → (T007 and T008 in parallel) → T009.
- **User Story 2 (Phase 4)**: Starts after T005; T010 → (T011 and T012 in parallel) → T013.
- **User Story 3 (Phase 5)**: Starts after T005; T014 → (T015 and T016 in parallel) → T017.
- **Polish (Phase 6)**: Starts after the selected story phases. T018, T019, and T020 can run in parallel; T021 follows relevant documentation edits; T022 follows T021; T023 and T024 can run in parallel; T025 follows the complete bundle; T026 is last.
- **Unified follow-up (Phase 7)**: Starts after T026; T027 → T028 → T029 → T030.
- **Actor-route follow-up (Phase 8)**: Starts after T030; T031 → T032 → T033 → T034.

### User Story Dependencies

- **US1 (P1)**: No dependency on another story. It uses only the shared projection and can ship as a hierarchy-only MVP.
- **US2 (P1)**: No dependency on US1. It uses shared access indexes and is independently testable against exact source relationships.
- **US3 (P2)**: No dependency on US1 or US2. It uses shared actor/access indexes to present complete routes and supporting context.
- All stories edit `tools/mlar-archify/bin/mlar-archify.mjs`; parallel story development therefore requires separate branches and coordinated integration even though their behavior is independently testable.

### Contract and Entity Mapping

- **Visualization input compatibility contract**: T003-T005 and regression task T023 preserve canonical v1 semantics.
- **Memory hierarchy window and memory layer node**: T006-T009 implement US1.
- **Actor access unit and memory access window**: T010-T013 implement US2.
- **Supporting context window and complete bundle contract**: T014-T017 implement US3.
- **Gallery catalog entry**: T008, T012, and T016 extend navigation incrementally without adding a renderer.

### Within Each User Story

- Add the story's failing acceptance tests first.
- Implement planner and gallery behavior against the documented contracts.
- Run the story's focused tests before proceeding to integration.
- Preserve canonical IDs and source relationships throughout; derived containment is never counted as access.

### Parallel Opportunities

- T007 and T008 can run together after T006.
- T011 and T012 can run together after T010.
- T015 and T016 can run together after T014.
- T018, T019, and T020 can run together after the selected stories are complete.
- T023 and T024 can run together after implementation and documentation settle.

---

## Parallel Example: User Story 1

```text
After T006:
Task T007: Implement hierarchy projection and bounded Archify views in tools/mlar-archify/bin/mlar-archify.mjs
Task T008: Implement hierarchy-first gallery catalog/navigation in tools/mlar-archify/lib/gallery.mjs
```

## Parallel Example: User Story 2

```text
After T010:
Task T011: Implement memory-anchored access grouping in tools/mlar-archify/bin/mlar-archify.mjs
Task T012: Implement memory-access catalog/navigation in tools/mlar-archify/lib/gallery.mjs
```

## Parallel Example: User Story 3

```text
After T014:
Task T015: Implement supporting-context planning in tools/mlar-archify/bin/mlar-archify.mjs
Task T016: Implement supporting-context gallery organization in tools/mlar-archify/lib/gallery.mjs
```

---

## Implementation Strategy

### MVP First

1. Complete Setup and Foundational phases.
2. Project hierarchy, recursive structure, and exact actor access from the same v1 source facts.
3. Emit one combined primary diagram when it fits the 12-node limit.
4. Retain hierarchy/access partitioning only as overflow for larger models.

### Incremental Delivery

1. **US1 + US2**: One diagram for memory hierarchy, recursive details, connected actors, and directional access when it fits.
2. **US3**: Cross-level routes in that diagram, with resource/network context remaining secondary.
3. **Overflow**: Deterministic bounded hierarchy/access pages only for over-limit models.
4. **Polish**: Canonical docs, renderer validation, compatibility regression, measured acceptance, and worktree review.

### Parallel Team Strategy

1. Complete T001-T005 together.
2. Develop US1, US2, and US3 on separate branches because each edits the planner file.
3. Within each story, implement planner and gallery tasks in parallel after its test task.
4. Integrate in priority order US1 → US2 → US3, rerunning focused tests after each merge.

## Notes

- `[P]` tasks edit different files and can proceed together only after their listed prerequisite.
- T021 review found both documentation diagrams already describe the adapter generically as the semantic-view planner and preserve the correct Rust → v1 YAML → adapter → Archify boundary. No JSON source changed, so diagram regeneration was not required.
- T024: all 11 specifications passed showcase validation and delivery. Visual checks were recorded as `skipped` for all 11 because Chrome/Chromium was unavailable; no visual pass is claimed.
- T025 machine-assisted journey over the generated manifest/specifications: hierarchy identification took 0.044 ms and found DRAM, replicated `x=8 × y=8` L1, both scope boundaries, and the L1 bank layer; SC-003 access identification took 0.092 ms and found `matrix_lane` and `vector_lane`, each with read and write edges; SC-004 route tracing took 0.052 ms and found both DRAM → `dram_l1_noc0` → L1 and L1 → `l1_dram_noc1` → DRAM on the first query. These results validate the artifact paths and are under the 60-second target, but they do not establish the specified 90% human-evaluator rate; no human study or browser visual review was performed.
- T030 follow-up: the rebuilt 2D mesh bundle has exactly one 9-node `memory_hierarchy` primary diagram, zero `memory_access` diagrams, all 10 canonical read/write relationships, both derived bank-containment relationships, and zero omitted source IDs. One machine-assisted query found hierarchy, both bank layers, both compute processors with read/write access, and both cross-level mover routes in that single diagram in 0.195 ms. All seven generated diagrams passed showcase validation/delivery; browser visual checks remain truthfully `skipped` because Chrome/Chromium is unavailable. Rust/v1 compatibility-boundary files retain zero diff.
- T034 follow-up: the rebuilt 2D mesh primary specification places DRAM at column 0, all five connected processors/data movers at column 1, and L1 at column 2. It contains DRAM → `dram_l1_noc0` → L1 and L1 → `l1_dram_noc1` → DRAM with arrow-only access segments and zero access labels. All 14 adapter tests pass, all seven diagrams pass showcase validation/delivery with zero source omissions, and all seven browser visual checks remain `skipped` because Chrome/Chromium is unavailable. Documentation typecheck/build, `cargo fmt --check`, and the four focused Rust visualization tests pass; the Rust projection, public re-export, v1 schema, and tracked visualization YAML retain zero diff.
- Rust and v1 schema files are regression boundaries, not implementation targets.
- Derived memory-layer IDs are presentation-only and cannot become processor/data-mover endpoints.
- Scope containment communicates ownership only; it never creates read/write access.
- Keep every source component and relationship discoverable and every diagram at or below 12 primary nodes.
- Never patch generated Archify HTML; edit its JSON source and regenerate.
- Report skipped browser or external-tool checks truthfully.
