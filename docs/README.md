# MLAR Documentation

The documentation is split by responsibility:

- Markdown files in this directory are the hand-authored textual documentation
  and remain the source for the Docusaurus documentation site.
- Archify JSON specifications also live directly in this directory. Each JSON
  file is the source of truth for its architecture visualization.
- [`.lavish/`](.lavish/) contains generated HTML review artifacts. This includes
  Archify's standalone architecture viewer and a review build of the complete
  Docusaurus documentation site. Do not edit these outputs by hand. Generated
  contents are ignored by Git and should be regenerated locally for review.
  Git tracks only `.lavish/.gitkeep` to preserve the empty directory in a fresh
  checkout.

## Text Documentation

- [Installation and toolchain setup](installation.md)
- [Basic architectural concepts](architecture-concepts.md)
- [Usage and end-to-end examples](usage.md)
- [Performance-model YAML](perf-yaml.md)
- [Software architecture and repository layout](software-architecture.md)

## Architecture Visualizations

- [High-level project architecture](.lavish/architecture/mlar-project-architecture.html)
  ([Archify JSON source](mlar-project-architecture.json))
- [Docusaurus documentation artifact](.lavish/docusaurus/index.html)

## Regenerating A Diagram

Run Archify from the repository root. Validate the JSON source before delivery:

```bash
node tools/archify/bin/archify.mjs validate architecture \
  docs/mlar-project-architecture.json \
  --quality showcase --repo-root . --json

node tools/archify/bin/archify.mjs deliver architecture \
  docs/mlar-project-architecture.json \
  docs/.lavish/architecture/mlar-project-architecture.html \
  --quality showcase --repo-root . --json

node tools/archify/bin/archify.mjs visual-check \
  docs/.lavish/architecture/mlar-project-architecture.html --json
```

All diagram changes must be made in the JSON specification and regenerated;
the HTML artifact is delivery output only.

Documentation diagrams must use the Archify skill. Embed a delivered diagram
in a Markdown/MDX page with the shared Docusaurus component:

```mdx
import ArchifyDiagram from '@site/src/components/ArchifyDiagram';

<ArchifyDiagram
  src="/diagrams/mlar-project-architecture.html"
  title="mlar-rust project architecture"
/>
```

Do not inline Archify's generated HTML, SVG, scripts, or styles in Markdown.
Both Docusaurus build commands copy delivered files from
`docs/.lavish/architecture/` into their `diagrams/` output, so Archify delivery
must run first.

## Regenerating The Docusaurus Artifact

The ordinary `npm run build` command remains the production browser-router
build for deployment. Generate the Lavish-compatible, hash-router artifact with:

```bash
cd docsite
npm run build:lavish
```

This writes the complete reviewable site—HTML, JavaScript, CSS, and images—to
`docs/.lavish/docusaurus/`. Treat the entire directory as one artifact and open
its `index.html` in Lavish. Embedded Archify viewers are copied into
`docs/.lavish/docusaurus/diagrams/` after the site build.

For page-level review, use the HTML produced by Docusaurus rather than creating
a second hand-written renderer. Preserve the page's emitted Docusaurus hierarchy
inside the artifact tree; for example:

```text
docsite/build/docs/usage/index.html
  -> docs/.lavish/docusaurus/docs/usage/index.html
```

The page's CSS, JavaScript, images, and other generated dependencies must move
with it. Feedback is applied to the corresponding Markdown file under `docs/`
or to `docsite/`, followed by another build; generated files under `.lavish/`
are never edited directly.
