# MLAR documentation site

This Docusaurus site renders the Markdown files in the repository's
`docs/` directory. The documentation is referenced directly, so there is no
second copy to keep synchronized. Archify JSON sources also live directly under
`docs/`; generated diagram HTML is written to the Git-ignored
`docsite/static/diagrams/` directory.

## Local development

From `docsite/`:

```bash
npm ci
npm start
```

The development server opens at `http://localhost:3000/loom-mlar/`.
The `prestart` script validates and delivers every Archify JSON specification
under `docs/`, then copies the generated HTML into `static/diagrams/` so the
development server can resolve embedded `/diagrams/...` URLs.

Create a production deployment build with:

```bash
npm run build
```

The static output is written to `docsite/build/`. The configured URL and base
path are ready for deployment to the `ecolab-nus/loom-mlar` GitHub Pages site.
The `prebuild` step recompiles the Archify diagrams from their JSON sources. The
Docusaurus static asset pipeline includes the generated files as
`build/diagrams/*.html`.

Use `npm run diagrams:build` to compile the diagrams without starting or
building Docusaurus. This command uses the vendored Archify CLI with showcase
validation and writes standalone HTML under `static/diagrams/`.
Docusaurus pages embed those files through `src/components/ArchifyDiagram.tsx`
using site-relative `/diagrams/...` URLs.
