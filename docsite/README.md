# MLAR documentation site

This directory contains the Docusaurus application for the MLAR documentation.
It reads its content and diagrams directly from the repository-level `docs/`
directory; there is no second documentation copy to synchronize.

## Documentation sources

- `docs/*.md` files are the hand-authored documentation pages.
- `docs/*.json` files are Archify diagram specifications. They are editable
  diagram source files, not generated Docusaurus data or general MLAR
  architecture instances.
- `docsite/static/diagrams/` contains generated standalone Archify HTML and is
  ignored by Git.

`npm start` and `npm run build` automatically run `npm run diagrams:build`.
That command discovers every `docs/*.json` file, reads its `diagram_type`, runs
vendored Archify showcase validation and delivery, and writes each result to:

```text
docsite/static/diagrams/<diagram-name>/index.html
```

Docusaurus pages embed these directory-style routes through
`src/components/ArchifyDiagram.tsx`, for example
`/diagrams/project-overview/`.

## Install dependencies

From this directory:

```bash
npm ci
```

Node.js 20 or newer is required.

## Development

Start the development server with hot reload:

```bash
npm start
```

Open `http://localhost:3000/loom-mlar/`. The `prestart` hook compiles all
Archify diagrams before Docusaurus starts. After editing a diagram JSON while
the server is running, regenerate it with `npm run diagrams:build` and refresh
the page, or restart `npm start`.

Run `npm run typecheck` when changing Docusaurus TypeScript or React code.

## Build and preview the deployment artifact

Create the complete production site with:

```bash
npm run build
```

The `prebuild` hook validates and compiles the Archify JSON diagrams before
Docusaurus writes the deployable site to `docsite/build/`. The generated
diagram pages are included as
`build/diagrams/<diagram-name>/index.html`.

Preview that exact artifact locally:

```bash
npm run serve
```

Open `http://localhost:3000/loom-mlar/docs/project-overview`. Stop any existing
`npm start` process first: development and preview servers both use port 3000
by default.

## Deployment

Deploy the entire `docsite/build/` directory to a static web host. Do not
publish an individual HTML file or only the `diagrams/` directory; the site also
requires its generated JavaScript, CSS, images, routes, and diagram assets.

The checked-in Docusaurus configuration targets
`https://ecolab-nus.github.io/loom-mlar/`, so GitHub Pages must publish the
contents of `build/` under the `/loom-mlar/` base path. This repository does not
currently contain a deployment workflow; the hosting or CI configuration is
responsible for running `npm ci`, `npm run build`, and uploading the complete
`build/` directory.

## Diagram-only build

To validate and compile the Archify JSON files without starting or building
Docusaurus, run:

```bash
npm run diagrams:build
```

All diagram edits belong in the corresponding `docs/*.json` source. Never edit
the generated HTML under `static/diagrams/`.
