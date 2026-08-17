# Vendored Archify Runtime

This directory contains the runtime files needed to validate and deliver
Archify diagrams reproducibly inside this repository.

- Upstream: `https://github.com/tt-a1i/archify`
- Version: `2.14.0`
- License: MIT; see [`LICENSE`](LICENSE)
- Local entry point: `node tools/archify/bin/archify.mjs`

The copy is intentionally project-relative. Do not replace it with a path into
an individual developer's Codex skill installation. Upgrade it as a standalone
reviewed change, preserving the upstream version, license, schemas, renderer
runtime, and validation behavior together.
