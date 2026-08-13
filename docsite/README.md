# MLAR documentation site

This Docusaurus site renders the Markdown files in the repository's
`docs/` directory. The documentation is referenced directly, so there is no
second copy to keep synchronized. Archify JSON sources also live directly under
`docs/`; generated architecture and Docusaurus review artifacts live under
`docs/.lavish/`.

## Local development

From `docsite/`:

```bash
npm install
npm start
```

The development server opens at `http://localhost:3000/loom-mlar/`.

Create a production deployment build with:

```bash
npm run build
```

The static output is written to `docsite/build/`. The configured URL and base
path are ready for deployment to the `ecolab-nus/loom-mlar` GitHub Pages site.
The `postbuild` step copies delivered Archify HTML from
`docs/.lavish/architecture/` into `build/diagrams/`.

Create a Lavish-compatible review artifact with:

```bash
npm run build:lavish
```

This uses a dedicated hash-router configuration with relative assets and writes
the complete static site to `docs/.lavish/docusaurus/`. Open its `index.html`
with Lavish; do not edit the generated files directly. The command also copies
the delivered Archify viewers into the artifact's `diagrams/` directory.

Before either build, use the Archify skill to validate the diagram JSON under
`docs/` and deliver its standalone HTML under
`docs/.lavish/architecture/`. Docusaurus pages embed those files through
`src/components/ArchifyDiagram.tsx` using site-relative `/diagrams/...` URLs.

When a browser-router production build is used for page-level review, preserve
each page's emitted path when mirroring it into the same artifact tree. For
example, `build/docs/usage/index.html` maps to
`docs/.lavish/docusaurus/docs/usage/index.html`. Mirror all referenced assets as
well; never copy an isolated HTML file that cannot render on its own.
