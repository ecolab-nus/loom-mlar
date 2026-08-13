# MLAR documentation site

This Docusaurus site renders the Markdown files in the repository's
`docs/text/` directory. The documentation is referenced directly, so there is
no second copy to keep synchronized. Archify sources live under
`docs/.archify/`; generated architecture and Docusaurus review artifacts live
under `docs/.lavish/`.

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

Create a Lavish-compatible review artifact with:

```bash
npm run build:lavish
```

This uses a dedicated hash-router configuration with relative assets and writes
the complete static site to `docs/.lavish/docusaurus/`. Open its `index.html`
with Lavish; do not edit the generated files directly.

When a browser-router production build is used for page-level review, preserve
each page's emitted path when mirroring it into the same artifact tree. For
example, `build/docs/usage/index.html` maps to
`docs/.lavish/docusaurus/docs/usage/index.html`. Mirror all referenced assets as
well; never copy an isolated HTML file that cannot render on its own.
