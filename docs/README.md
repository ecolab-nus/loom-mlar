# MLAR Documentation

The documentation is split by responsibility:

- Markdown files in this directory are the hand-authored textual documentation
  and remain the source for the Docusaurus documentation site.
- Archify JSON specifications also live directly in this directory. Each JSON
  file is the source of truth for its architecture visualization.
- Generated diagram HTML is written to `docsite/static/diagrams/` by the
  documentation build and is ignored by Git.

## Text Documentation

- [Project overview](project-overview.md)
- [Installation and toolchain setup](installation.md)
- [Basic architectural concepts](architecture-concepts.md)
- [Usage and end-to-end examples](usage.md)
- [Performance-model YAML](perf-yaml.md)
- [Software architecture and repository layout](software-architecture.md)

## Architecture Visualizations

- [Project overview and artifact flow](project-overview.md)
  ([Archify JSON source](project-overview.json))
- [High-level project architecture](software-architecture.md)
  ([Archify JSON source](mlar-project-architecture.json))

## Regenerating A Diagram

The documentation build validates and compiles every `docs/*.json` Archify
source automatically:

```bash
cd docsite
npm run diagrams:build
```

The generated standalone HTML files are placed under
`docsite/static/diagrams/`. All diagram changes must be made in the JSON
specification; generated HTML is build output only.

`npm start` and `npm run build` under `docsite/` automatically run this
validation and delivery workflow before Docusaurus starts compiling. Run
`npm run diagrams:build` there to regenerate the diagrams without starting the
site.

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
Docusaurus copies `static/diagrams/` into the production site automatically.

## Building The Docusaurus Site

Create the production static site with:

```bash
cd docsite
npm run build
```

The `prebuild` hook compiles the diagrams before Docusaurus builds the complete
site under `docsite/build/`, including `build/diagrams/*.html`.
