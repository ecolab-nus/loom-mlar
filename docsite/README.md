# MLAR documentation site

This Docusaurus site renders the Markdown files in the repository's top-level
`docs/` directory. The documentation is referenced directly, so there is no
second copy to keep synchronized.

## Local development

From `docsite/`:

```bash
npm install
npm start
```

The development server opens at `http://localhost:3000/loom-mlar/`.

Create a production build with:

```bash
npm run build
```

The static output is written to `docsite/build/`. The configured URL and base
path are ready for deployment to the `ecolab-nus/loom-mlar` GitHub Pages site.

