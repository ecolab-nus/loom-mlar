# MLAR Documentation

The documentation is split by responsibility:

- [`text/`](text/) contains the hand-authored Markdown documentation. These
  pages remain the source for the Docusaurus documentation site.
- [`.archify/`](.archify/) contains Archify JSON specifications. Each JSON file
  is the source of truth for its architecture visualization.
- [`.lavish/`](.lavish/) contains generated HTML review artifacts. This includes
  Archify's standalone architecture viewer and a review build of the complete
  Docusaurus documentation site. Do not edit these outputs by hand. Generated
  contents are ignored by Git and should be regenerated locally for review.
  Git tracks only `.lavish/.gitkeep` to preserve the empty directory in a fresh
  checkout.

## Text Documentation

- [Installation and toolchain setup](text/installation.md)
- [Basic architectural concepts](text/architecture-concepts.md)
- [Usage and end-to-end examples](text/usage.md)
- [Performance-model YAML](text/perf-yaml.md)
- [Software architecture and repository layout](text/software-architecture.md)

## Architecture Visualizations

- [High-level project architecture](.lavish/architecture/mlar-project-architecture.html)
  ([Archify JSON source](.archify/mlar-project-architecture.json))
- [Docusaurus documentation artifact](.lavish/docusaurus/index.html)

## Regenerating A Diagram

Run Archify from the repository root. Validate the JSON source before delivery:

```bash
node /path/to/archify/bin/archify.mjs validate architecture \
  docs/.archify/mlar-project-architecture.json \
  --quality showcase --repo-root . --json

node /path/to/archify/bin/archify.mjs deliver architecture \
  docs/.archify/mlar-project-architecture.json \
  docs/.lavish/architecture/mlar-project-architecture.html \
  --quality showcase --repo-root . --json
```

All diagram changes must be made in the JSON specification and regenerated;
the HTML artifact is delivery output only.

## Regenerating The Docusaurus Artifact

The ordinary `npm run build` command remains the production browser-router
build for deployment. Generate the Lavish-compatible, hash-router artifact with:

```bash
cd docsite
npm run build:lavish
```

This writes the complete reviewable site—HTML, JavaScript, CSS, and images—to
`docs/.lavish/docusaurus/`. Treat the entire directory as one artifact and open
its `index.html` in Lavish.

For page-level review, use the HTML produced by Docusaurus rather than creating
a second hand-written renderer. Preserve the page's emitted Docusaurus hierarchy
inside the artifact tree; for example:

```text
docsite/build/docs/usage/index.html
  -> docs/.lavish/docusaurus/docs/usage/index.html
```

The page's CSS, JavaScript, images, and other generated dependencies must move
with it. Feedback is applied to `docs/text/` or `docsite/`, followed by another
build; generated files under `.lavish/` are never edited directly.
