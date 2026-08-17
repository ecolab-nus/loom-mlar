<!--
Sync Impact Report
- Version change: 1.2.0 -> 1.3.0
- Modified principles: none
- Added principles: VI. Occam's Razor; VII. Human-Readable Textual Contracts
- Modified sections: Core Principles (expanded from five to seven principles)
- Added sections: none
- Removed sections: none
- Follow-up TODOs: none
-->
# MLAR Rust Constitution

## Project Purpose and Scope

MLAR Rust provides a shared abstraction between hardware architecture design and compiler-driven
performance evaluation for processors and accelerators. The abstraction targets cycle-approximate
precision: it MUST describe an architecture precisely enough for fast pre-silicon performance
estimation, but it does not claim cycle-accurate simulation.

The project has two primary objectives:

1. **Architecture design and exploration.** MLAR gives hardware designers an abstraction for
   specifying processor or accelerator architectures, and allow designers to visualize, modify the design. The
   specification MUST express the functional, structural, and performance information needed for
   cycle-approximate analysis. This includes characteristics such as throughput, bandwidth,
   connectivity, and topology without fixing the project to a permanent catalog of architecture
   concepts. Performance information support symbolic models and relationships.
2. **Compiler communication and mapping-aware evaluation.** MLAR provides the compiler with the
   architectural, functional, and performance information needed to map workloads and evaluate each
   mapping or schedule. Because realized performance depends on the selected workload mapping, MLAR 
   enables estimates that account for both the workload and the compiler's mapping decisions,
   rather than reporting architecture-only peak figures.

MLAR is therefore the information contract that connects architecture exploration to compiler
mapping and evaluation. It does not replace the compiler, and it is not a cycle-accurate simulator.
Its value is fast, sufficiently precise, workload-aware specification and estimation during pre-silicon design.

## Core Principles

### I. Preserve Library Contracts and Model Semantics
MLAR Rust MUST remain a library-first implementation with its common public API exposed through
`src/lib.rs`; a top-level CLI MUST NOT be introduced without an explicit product decision. Changes
to architecture, schedule, or other serialized public structures MUST preserve serde compatibility
or provide an intentional, documented migration. Visualization schema-version strings and
`schemas/mlar-visualization-v1.schema.json` are compatibility boundaries. Checked and unchecked
MLIR export MUST preserve their existing concretization and processor-source requirements, and
checked export MUST retain validator and experimental-feature enforcement. Schedule evaluation
MUST retain its defined recursive resolution, sequential sum, parallel maximum, Cartesian scenario
composition, constraint conjunction, and overlapping-scenario behavior unless a separately approved
specification changes that contract. These rules prevent apparently local edits from silently
breaking downstream users or changing model meaning.

### II. Verify Changes with Proportionate Evidence
Every change to parsing, symbolic math, schedule evaluation, architecture or network construction,
MLIR export, or visualization payloads MUST add or update focused tests. Rust changes MUST be
formatted with `cargo fmt` and finish with `cargo fmt --check`. Validation MUST begin with checks
focused on the changed area and expand in proportion to risk and available dependencies. Viewer
changes MUST run the relevant Node tests or build; documentation-site changes MUST run typechecking
and the relevant Docusaurus build. A skipped check, unavailable external validator, missing browser,
or dependency-blocked command MUST be reported as such and MUST NOT be described as passing.
Rationale: trustworthy evidence is more important than an unqualified green summary.

### III. Keep Visualization Semantics Complete and Bounded
The Rust projection, versioned schema, tracked normalized YAML sample, adapter tests, and relevant
documentation MUST change together whenever the visualization payload changes. Every source
component and relationship MUST appear in at least one generated semantic view. Replicated scopes
MUST remain dimension and instance-count metadata; implementations MUST NOT expand mesh tiles or
collapse distinct model entities into synthetic nodes. A semantic diagram MUST contain no more than
12 primary nodes, with complex models split into multiple views. Project-authored documentation,
CLI output, labels, and gallery UI MUST be English-only. The gallery MAY organize, filter, and embed
diagrams, but MUST NOT implement a second renderer. These constraints preserve traceability and
readability without changing architecture semantics for presentation.

### IV. Treat Documentation Sources as Reproducible Artifacts
User-visible behavior, exported schemas, and typical workflows MUST be reflected in the relevant
hand-authored Markdown and examples, aligned with the public API. `docs/*.md` and `docsite/` are the
editable textual sources; separate hand-authored HTML versions MUST NOT be created. Architecture,
workflow, sequence, data-flow, and lifecycle diagrams MUST use Archify JSON sources directly under
`docs/` unless the user explicitly requests another format. Diagram delivery MUST use the vendored,
project-relative Archify CLI at showcase quality, finish with all nine artifact checks, zero
composition errors, and zero warnings, and run `visual-check` on the delivered HTML. Generated HTML
MUST NOT be patched. Docusaurus MUST embed delivered diagrams through `ArchifyDiagram.tsx`, and a
documentation build is incomplete when an embedded diagram or any referenced site asset is missing.
Rationale: tracked sources must be sufficient to reproduce the complete published documentation.

### V. Protect Source Truth and User Work
Agents and contributors MUST inspect relevant existing code and documentation before modifying
them, prefer established builder-style APIs, and resolve material design ambiguity with the user.
Generated artifacts MUST remain in their designated ignored locations, except for the tracked MLIR
and visualization samples intentionally rewritten by focused integration tests. Before completion,
`git status --short` and relevant diffs MUST be inspected; unexpected changes MUST be investigated
and MUST NOT be silently discarded, overwritten, or included. Generated results and external-tool
limitations MUST be reported accurately. This principle protects both canonical sources and
unrelated work already present in a shared working tree.

### VI. Apply Occam's Razor
Implementations MUST use the simplest mechanism that satisfies verified requirements. Every new
abstraction, branch, state variable, cache, fallback, and feature flag MUST be justified by a
concrete requirement and challenged against a smaller design. Simplicity means the fewest justified
mechanisms, not the fewest lines of code. A review MUST raise a simplicity issue only when it can
identify a smaller concrete design and show that the unnecessary mechanism adds correctness, drift,
testing, or ownership cost. Preference or brevity alone is not sufficient grounds for an issue.
Rationale: eliminating unjustified mechanisms reduces failure modes and long-term maintenance cost
without rewarding compressed or obscure code.

### VII. Use Human-Readable Textual Contracts
All information exposed by the core MLAR specification framework or by adjacent capabilities,
including visualization, MUST have a canonical human-readable textual file representation. MLIR,
JSON, and YAML are representative formats; the specific syntax MAY evolve with the governed
contract. Components MUST exchange information through documented, versioned textual files rather
than depend on private in-memory coupling or opaque binary-only formats. A component MAY use typed
in-memory structures internally, but those structures MUST NOT be the sole inter-component contract.
Developers MUST be able to inspect the meaningful content of exchanged artifacts directly with
general-purpose text tools. Rationale: textual file boundaries make system behavior reviewable,
debuggable, reproducible, and independently consumable by architecture, compiler, and visualization
tooling.

## Engineering Constraints

- The crate MUST target Rust edition 2024 and a compatible stable Rust toolchain.
- The visualization adapter and documentation site MUST use Node.js 20 or newer. Clean installs
  MUST use their lockfiles through `npm ci`.
- `adl-opt` and `loom-opt` absence MAY produce non-blocking build warnings, but checked MLIR export
  MUST surface `MlirExportError::ToolNotFound`; tests requiring those tools MAY skip only when the
  skip is reported truthfully.
- Existing builders such as `FuncPerfModel::builder()`, `ComputeProcessor::builder()`,
  `DataMover::builder()`, `ScaleOutNetwork::mesh(...)`, and fluent `Architecture::with_*` methods
  MUST be preferred when they cover the requested construction.
- The vendored `tools/archify/` CLI MUST be invoked through its project-relative path. Repository
  files MUST NOT contain machine-specific skill paths or depend on global Archify installation.
- Tracked sources MUST remain sufficient to regenerate ignored visualization output, delivered
  diagrams, and the complete Docusaurus build tree.

## Development Workflow and Quality Gates

1. Read `README.md`, `docs/software-architecture.md`, `tests/2d_mesh/arch.rs`, and the directly
   affected implementation before making broad architectural changes.
2. Confirm any unclear or materially consequential design choice before editing. For an approved
   change, use existing APIs and preserve the contracts defined by this constitution.
3. Add or update focused coverage and documentation in the same change. Schema changes MUST update
   Rust types, schema, tracked sample, adapter tests, and documentation as one compatibility unit.
4. Run the narrowest relevant checks first, then broader checks warranted by risk. Rust work MUST
   finish with formatting verification; visualization and docsite work MUST finish with their
   required Node, Archify, and Docusaurus checks.
5. When documentation diagrams change, edit only the JSON source, then repeat Archify validation,
   delivery, and visual checking. An exit code indicating a browser-based check was skipped MUST be
   recorded as skipped.
6. Inspect the final worktree and all task-related diffs. Explicitly distinguish expected tracked
   sample rewrites, ignored generated output, pre-existing user changes, passed checks, and skipped
   or blocked checks in the handoff.

## Governance

This constitution is the highest-level engineering policy for MLAR Rust. `AGENTS.md` supplies
operational repository guidance and command details; when it conflicts with this constitution, the
constitution governs. Reviews and implementation handoffs MUST verify compliance with the applicable
principles and quality gates. Any exception MUST be explicit, scoped, justified, and approved by a
project maintainer before merge.

Amendments MUST be proposed as a change to this file with a written rationale, a Sync Impact Report,
and any required migration or compatibility plan. Maintainer approval is required before an
amendment takes effect. The amendment MUST update the Last Amended date and version according to
semantic versioning: MAJOR for incompatible governance changes or principle removals/redefinitions,
MINOR for new principles or materially expanded obligations, and PATCH for non-semantic
clarifications. Compliance review MUST occur when planning a feature, during code review, and before
declaring implementation complete.

**Version**: 1.3.0 | **Ratified**: 2026-08-17 | **Last Amended**: 2026-08-17
