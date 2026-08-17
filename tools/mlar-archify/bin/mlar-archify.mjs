#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';

import Ajv2020 from 'ajv/dist/2020.js';
import YAML from 'yaml';

import { renderGalleryHtml } from '../lib/gallery.mjs';

const toolRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = path.resolve(toolRoot, '..', '..');
const schemaPath = path.join(repoRoot, 'schemas', 'mlar-visualization-v1.schema.json');
const archifyBin = path.join(repoRoot, 'tools', 'archify', 'bin', 'archify.mjs');
const MAX_COMPONENTS = 12;

function usage() {
  process.stderr.write(
    [
      'Usage:',
      '  mlar-archify build <input.yaml> <output-directory> [--visual-check]',
      '  mlar-archify serve <output-directory> [--host 127.0.0.1] [--port 4173]',
      '',
    ].join('\n'),
  );
}

function fail(message, details = undefined) {
  const error = { status: 'error', error: message };
  if (details !== undefined) error.details = details;
  process.stderr.write(`${JSON.stringify(error, null, 2)}\n`);
  process.exit(1);
}

function parseArgs(argv) {
  if (argv[0] === 'build') return parseBuildArgs(argv);
  if (argv[0] === 'serve') return parseServeArgs(argv);
  usage();
  process.exit(2);
}

function parseBuildArgs(argv) {
  if (!argv[1] || !argv[2]) {
    usage();
    process.exit(2);
  }
  let visualCheck = false;
  for (let index = 3; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--visual-check') {
      visualCheck = true;
      continue;
    }
    fail(`Unknown option '${argument}'.`);
  }
  return {
    command: 'build',
    inputPath: path.resolve(argv[1]),
    outputDirectory: path.resolve(argv[2]),
    visualCheck,
  };
}

function parseServeArgs(argv) {
  if (!argv[1]) {
    usage();
    process.exit(2);
  }
  let host = '127.0.0.1';
  let port = 4173;
  for (let index = 2; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--host') {
      host = argv[index + 1];
      index += 1;
      if (!host) fail('--host requires a value.');
      continue;
    }
    if (argument === '--port') {
      port = Number(argv[index + 1]);
      index += 1;
      if (!Number.isInteger(port) || port < 1 || port > 65535) {
        fail('--port must be an integer between 1 and 65535.');
      }
      continue;
    }
    fail(`Unknown option '${argument}'.`);
  }
  return { command: 'serve', outputDirectory: path.resolve(argv[1]), host, port };
}

function loadDocument(inputPath) {
  let source;
  try {
    source = fs.readFileSync(inputPath, 'utf8');
  } catch (error) {
    fail(`Could not read '${inputPath}': ${error.message}`);
  }

  let document;
  try {
    document = YAML.parse(source);
  } catch (error) {
    fail(`Could not parse visualization YAML: ${error.message}`);
  }

  const schema = JSON.parse(fs.readFileSync(schemaPath, 'utf8'));
  const ajv = new Ajv2020({ allErrors: true, strict: true });
  const validate = ajv.compile(schema);
  if (!validate(document)) {
    fail('Visualization YAML does not satisfy mlar.visualization.v1.', validate.errors);
  }
  try {
    validateReferences(document);
  } catch (error) {
    fail(error.message);
  }
  return { document, source };
}

export function validateReferences(document) {
  const scopeIds = new Set(document.scopes.map((scope) => scope.id));
  const componentIds = new Set(document.components.map((component) => component.id));
  const allIds = new Set();
  const relationshipIds = document.relationships.map((relationship) => relationship.id);
  const linkIds = document.components
    .filter((component) => component.kind === 'network')
    .flatMap((component) => component.links.map((link) => link.id));
  for (const id of [
    document.architecture.id,
    ...scopeIds,
    ...componentIds,
    ...relationshipIds,
    ...linkIds,
  ]) {
    if (allIds.has(id)) throw new Error(`Duplicate canonical id '${id}'.`);
    allIds.add(id);
  }
  if (!scopeIds.has(document.architecture.root_scope)) {
    throw new Error(`Unknown root scope '${document.architecture.root_scope}'.`);
  }
  const parentByScope = new Map();
  for (const scope of document.scopes) {
    if (scope.parent_scope && !scopeIds.has(scope.parent_scope)) {
      throw new Error(`Scope '${scope.id}' references unknown parent '${scope.parent_scope}'.`);
    }
    parentByScope.set(scope.id, scope.parent_scope ?? null);
  }
  if (parentByScope.get(document.architecture.root_scope) !== null) {
    throw new Error(`Root scope '${document.architecture.root_scope}' must not have a parent.`);
  }
  for (const scope of document.scopes) {
    const visited = new Set();
    let cursor = scope.id;
    while (cursor !== document.architecture.root_scope) {
      if (visited.has(cursor)) throw new Error(`Scope parent cycle includes '${cursor}'.`);
      visited.add(cursor);
      cursor = parentByScope.get(cursor);
      if (cursor == null) {
        throw new Error(`Scope '${scope.id}' is not connected to the root scope.`);
      }
    }
  }
  for (const component of document.components) {
    if (!scopeIds.has(component.scope)) {
      throw new Error(`Component '${component.id}' references unknown scope '${component.scope}'.`);
    }
  }
  for (const relationship of document.relationships) {
    if (!componentIds.has(relationship.source) || !componentIds.has(relationship.target)) {
      throw new Error(`Relationship '${relationship.id}' has an unknown endpoint.`);
    }
  }
}

function dimensionsLabel(dimensions = []) {
  if (dimensions.length === 0) return '';
  return dimensions.map((dimension) => `${dimension.name}=${dimension.size.text}`).join(' × ');
}

function formatBytes(value) {
  if (value == null) return null;
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let amount = value;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return `${Number.isInteger(amount) ? amount : amount.toFixed(1)} ${units[unit]}`;
}

function formatCompactNumber(value) {
  return new Intl.NumberFormat('en', {
    notation: 'compact',
    maximumFractionDigits: 1,
  }).format(value);
}

function archifyComponent(component, index, columns) {
  const base = {
    id: component.id,
    label: component.name,
    row: Math.floor(index / columns),
    col: index % columns,
  };
  switch (component.kind) {
    case 'memory': {
      const details = [dimensionsLabel(component.dimensions), formatBytes(component.total_size_bytes)]
        .filter(Boolean)
        .join(' · ');
      return { ...base, type: 'database', sublabel: details || 'memory' };
    }
    case 'processor':
      return {
        ...base,
        type: 'backend',
        sublabel: `${component.functions.length} functions`,
        tag: component.effect,
      };
    case 'data_mover':
      return {
        ...base,
        type: 'messagebus',
        sublabel: `${component.functions.length} transfer functions`,
        tag: component.effect,
      };
    case 'resource':
      return {
        ...base,
        type: 'cloud',
        sublabel:
          component.capacity == null
            ? 'shared resource'
            : `capacity ${formatCompactNumber(component.capacity)}`,
        tag:
          component.resource_kind === 'exclusive'
            ? 'exclusive'
            : 'quantitative',
      };
    case 'network':
      return {
        ...base,
        type: 'messagebus',
        sublabel: `${component.network_kind} · bw=${component.bandwidth.text}`,
        tag: dimensionsLabel(component.dimensions) || undefined,
      };
    case 'scope': {
      const scopeDetails = [
        dimensionsLabel(component.dimensions),
        component.replication_factor > 1
          ? `${component.replication_factor} instances`
          : null,
      ]
        .filter(Boolean)
        .join(' · ');
      return {
        ...base,
        type: 'cloud',
        sublabel: scopeDetails || 'architecture scope',
      };
    }
    default:
      fail(`Unsupported component kind '${component.kind}'.`);
  }
}

function archifyConnection(relationship) {
  const labels = {
    read: 'read',
    write: 'write',
    requires: 'requires',
    network_attachment: 'attaches',
  };
  const connection = {
    id: relationship.id,
    from: relationship.source,
    to: relationship.target,
    label: labels[relationship.kind] ?? relationship.label,
  };
  if (relationship.kind === 'requires') connection.variant = 'dashed';
  if (relationship.kind === 'requires') connection.labelDy = 24;
  if (relationship.kind === 'read' || relationship.kind === 'write') connection.variant = 'emphasis';
  return connection;
}

function diagramId(parts) {
  return parts
    .join('-')
    .toLowerCase()
    .replace(/[^a-z0-9-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/-+/g, '-');
}

function createDiagram({
  id,
  title,
  section = 'other',
  primaryScopeId = null,
  components,
  relationships,
}) {
  const sortedComponents = [...components].sort((left, right) => left.id.localeCompare(right.id));
  if (sortedComponents.length === 0) return null;
  if (sortedComponents.length > MAX_COMPONENTS) {
    fail(`Diagram '${id}' has ${sortedComponents.length} components; maximum is ${MAX_COMPONENTS}.`);
  }
  const componentIds = new Set(sortedComponents.map((component) => component.id));
  const selectedRelationships = relationships
    .filter(
      (relationship) => componentIds.has(relationship.source) && componentIds.has(relationship.target),
    )
    .sort((left, right) => left.id.localeCompare(right.id));
  const columns = Math.min(4, Math.max(1, Math.ceil(Math.sqrt(sortedComponents.length))));
  const spec = {
    schema_version: 1,
    diagram_type: 'architecture',
    meta: {
      title,
      quality_profile: 'showcase',
    },
    layout: {
      mode: 'grid',
      origin: [40, 70],
      cols: columns,
      gapX: 86,
      gapY: 76,
      cellW: 176,
      cellH: 76,
    },
    components: sortedComponents.map((component, index) =>
      archifyComponent(component, index, columns),
    ),
    connections: selectedRelationships.map((relationship) =>
      archifyConnection(relationship),
    ),
  };
  return {
    id,
    title,
    type: 'architecture',
    section,
    primaryScopeId,
    spec,
    componentIds: sortedComponents.filter((component) => component.kind !== 'scope').map((component) => component.id),
    scopeIds: sortedComponents.filter((component) => component.kind === 'scope').map((component) => component.id),
    relationshipIds: selectedRelationships.map((relationship) => relationship.id),
  };
}

function chunks(values, size) {
  const result = [];
  for (let index = 0; index < values.length; index += size) {
    result.push(values.slice(index, index + size));
  }
  return result;
}

function relationDiagrams({ document, relationships, prefix, title, section }) {
  if (relationships.length === 0) return [];
  const byId = new Map(document.components.map((component) => [component.id, component]));
  const grouped = new Map();
  for (const relationship of relationships) {
    const source = byId.get(relationship.source);
    const target = byId.get(relationship.target);
    const anchor = source.kind === 'memory' || source.kind === 'resource' ? target : source;
    if (!grouped.has(anchor.id)) grouped.set(anchor.id, []);
    grouped.get(anchor.id).push(relationship);
  }
  const diagrams = [];
  for (const [anchorId, entries] of [...grouped.entries()].sort()) {
    const anchor = byId.get(anchorId);
    for (const [chunkIndex, relationChunk] of chunks(entries, 1).entries()) {
      const ids = new Set([anchorId]);
      for (const relationship of relationChunk) {
        ids.add(relationship.source);
        ids.add(relationship.target);
      }
      const suffix = entries.length > 1 ? `-${chunkIndex + 1}` : '';
      const diagram = createDiagram({
        id: diagramId([prefix, anchor.name, suffix]),
        title: `${title} · ${anchor.name}${suffix}`,
        section,
        primaryScopeId: anchor.scope,
        components: [...ids].map((id) => byId.get(id)),
        relationships: relationChunk,
      });
      if (diagram) diagrams.push(diagram);
    }
  }
  return diagrams;
}

export function planDiagrams(document) {
  const componentsByScope = new Map();
  for (const component of document.components) {
    if (!componentsByScope.has(component.scope)) componentsByScope.set(component.scope, []);
    componentsByScope.get(component.scope).push(component);
  }
  const childrenByScope = new Map();
  for (const scope of document.scopes) {
    if (!scope.parent_scope) continue;
    if (!childrenByScope.has(scope.parent_scope)) childrenByScope.set(scope.parent_scope, []);
    childrenByScope.get(scope.parent_scope).push(scope);
  }

  const diagrams = [];
  const orderedScopes = [...document.scopes].sort((left, right) => {
    if (left.id === document.architecture.root_scope) return -1;
    if (right.id === document.architecture.root_scope) return 1;
    return left.id.localeCompare(right.id);
  });
  for (const scope of orderedScopes) {
    const direct = (componentsByScope.get(scope.id) ?? []).filter(
      (component) => component.kind !== 'resource' && component.kind !== 'network',
    );
    const childScopes = (childrenByScope.get(scope.id) ?? []).map((child) => ({
      kind: 'scope',
      id: child.id,
      name: child.name,
      dimensions: child.dimensions,
      replication_factor: child.replication_factor,
    }));
    const scopeComponents = [...direct, ...childScopes];
    const scopeMarker = {
      kind: 'scope',
      id: scope.id,
      name: scope.name,
      dimensions: scope.dimensions,
      replication_factor: scope.replication_factor,
    };
    const componentChunks =
      scopeComponents.length === 0 ? [[]] : chunks(scopeComponents, MAX_COMPONENTS - 1);
    for (const [chunkIndex, componentChunk] of componentChunks.entries()) {
      const suffix = scopeComponents.length > MAX_COMPONENTS - 1 ? `-${chunkIndex + 1}` : '';
      const diagram = createDiagram({
        id: diagramId(['scope', scope.name, suffix]),
        title: `${scope.name} · components${suffix}`,
        section: 'overview',
        primaryScopeId: scope.id,
        components: [scopeMarker, ...componentChunk],
        relationships: [],
      });
      if (diagram) diagrams.push(diagram);
    }
  }

  diagrams.push(
    ...relationDiagrams({
      document,
      relationships: document.relationships.filter(
        (relationship) => relationship.kind === 'read',
      ),
      prefix: 'memory-reads',
      title: 'Memory reads',
      section: 'memory_reads',
    }),
  );
  diagrams.push(
    ...relationDiagrams({
      document,
      relationships: document.relationships.filter(
        (relationship) => relationship.kind === 'write',
      ),
      prefix: 'memory-writes',
      title: 'Memory writes',
      section: 'memory_writes',
    }),
  );
  diagrams.push(
    ...relationDiagrams({
      document,
      relationships: document.relationships.filter(
        (relationship) => relationship.kind === 'requires',
      ),
      prefix: 'resource-dependencies',
      title: 'Resource dependencies',
      section: 'resources',
    }),
  );
  diagrams.push(
    ...relationDiagrams({
      document,
      relationships: document.relationships.filter(
        (relationship) => relationship.kind === 'network_attachment',
      ),
      prefix: 'network-attachments',
      title: 'Network attachments',
      section: 'networks',
    }),
  );

  const coveredComponents = new Set(diagrams.flatMap((diagram) => diagram.componentIds));
  const uncovered = document.components.filter((component) => !coveredComponents.has(component.id));
  for (const [chunkIndex, componentChunk] of chunks(uncovered, MAX_COMPONENTS).entries()) {
    const diagram = createDiagram({
      id: diagramId(['other-components', String(chunkIndex + 1)]),
      title: `Other components · ${chunkIndex + 1}`,
      section: 'other',
      primaryScopeId: componentChunk[0]?.scope ?? null,
      components: componentChunk,
      relationships: document.relationships,
    });
    if (diagram) diagrams.push(diagram);
  }

  const ids = new Set();
  for (const diagram of diagrams) {
    if (ids.has(diagram.id)) fail(`Duplicate diagram id '${diagram.id}'.`);
    ids.add(diagram.id);
  }
  return diagrams;
}

function runArchify(args) {
  const result = spawnSync(process.execPath, [archifyBin, ...args], {
    cwd: repoRoot,
    encoding: 'utf8',
  });
  if (result.status !== 0) {
    fail(`Archify command failed: ${args.join(' ')}`, {
      status: result.status,
      stdout: result.stdout,
      stderr: result.stderr,
    });
  }
  const output = result.stdout.trim();
  try {
    return JSON.parse(output);
  } catch {
    return { raw: output };
  }
}

function writeJson(filePath, value) {
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function build({ inputPath, outputDirectory, visualCheck }) {
  const { document, source } = loadDocument(inputPath);
  const diagrams = planDiagrams(document);
  if (diagrams.length === 0) fail('The visualization document produced no diagrams.');

  const specificationsDirectory = path.join(outputDirectory, 'specifications');
  const htmlDirectory = path.join(outputDirectory, 'html');
  fs.mkdirSync(specificationsDirectory, { recursive: true });
  fs.mkdirSync(htmlDirectory, { recursive: true });

  const manifestDiagrams = [];
  for (const diagram of diagrams) {
    const specificationPath = path.join(specificationsDirectory, `${diagram.id}.json`);
    const htmlPath = path.join(htmlDirectory, `${diagram.id}.html`);
    writeJson(specificationPath, diagram.spec);
    const validation = runArchify([
      'validate',
      'architecture',
      specificationPath,
      '--quality',
      'showcase',
      '--json',
    ]);
    const delivery = runArchify([
      'deliver',
      'architecture',
      specificationPath,
      htmlPath,
      '--quality',
      'showcase',
      '--json',
    ]);
    const visual = visualCheck
      ? runArchify(['visual-check', htmlPath, '--json'])
      : { status: 'not_requested' };
    manifestDiagrams.push({
      id: diagram.id,
      title: diagram.title,
      type: diagram.type,
      section: diagram.section,
      primary_scope_id: diagram.primaryScopeId,
      specification: path.relative(outputDirectory, specificationPath),
      html: path.relative(outputDirectory, htmlPath),
      component_ids: diagram.componentIds,
      scope_ids: diagram.scopeIds,
      relationship_ids: diagram.relationshipIds,
      validation,
      delivery,
      visual_check: visual,
    });
  }

  const coveredComponents = new Set(
    manifestDiagrams.flatMap((diagram) => diagram.component_ids),
  );
  const coveredScopes = new Set(manifestDiagrams.flatMap((diagram) => diagram.scope_ids));
  const coveredRelationships = new Set(
    manifestDiagrams.flatMap((diagram) => diagram.relationship_ids),
  );
  const report = {
    schema_version: 'mlar.archify-conversion-report.v1',
    included_scope_ids: [...coveredScopes].sort(),
    omitted_scope_ids: document.scopes
      .map((scope) => scope.id)
      .filter((id) => !coveredScopes.has(id))
      .sort(),
    included_component_ids: [...coveredComponents].sort(),
    omitted_component_ids: document.components
      .map((component) => component.id)
      .filter((id) => !coveredComponents.has(id))
      .sort(),
    included_relationship_ids: [...coveredRelationships].sort(),
    omitted_relationship_ids: document.relationships
      .map((relationship) => relationship.id)
      .filter((id) => !coveredRelationships.has(id))
      .sort(),
    policy: {
      maximum_primary_nodes: MAX_COMPONENTS,
      expand_replicated_instances: false,
      create_synthetic_collapsed_nodes: false,
      partition_strategy: 'semantic_views_then_anchor_subgraphs',
    },
  };
  writeJson(path.join(outputDirectory, 'conversion-report.json'), report);
  if (
    report.omitted_scope_ids.length > 0 ||
    report.omitted_component_ids.length > 0 ||
    report.omitted_relationship_ids.length > 0
  ) {
    fail('Conversion coverage is incomplete.', report);
  }

  const manifest = {
    schema_version: 'mlar.archify-bundle.v1',
    source: {
      path: path.relative(repoRoot, inputPath),
      schema_version: document.schema_version,
      sha256: crypto.createHash('sha256').update(source).digest('hex'),
    },
    archify: {
      version: '2.14.0',
      entrypoint: path.relative(repoRoot, archifyBin),
      quality: 'showcase',
    },
    application: {
      entrypoint: 'index.html',
      mode: 'static-archify-gallery',
      language: 'en',
    },
    diagrams: manifestDiagrams,
  };
  writeJson(path.join(outputDirectory, 'bundle-manifest.json'), manifest);
  fs.writeFileSync(
    path.join(outputDirectory, 'index.html'),
    renderGalleryHtml(document, diagrams),
  );
  process.stdout.write(
    `${JSON.stringify(
      {
        status: 'ok',
        output: outputDirectory,
        application: path.join(outputDirectory, 'index.html'),
        diagrams: diagrams.length,
        serve: `node tools/mlar-archify/bin/mlar-archify.mjs serve ${path.relative(repoRoot, outputDirectory)}`,
      },
      null,
      2,
    )}\n`,
  );
}

const MIME_TYPES = new Map([
  ['.html', 'text/html; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.svg', 'image/svg+xml'],
  ['.png', 'image/png'],
  ['.webp', 'image/webp'],
  ['.jpg', 'image/jpeg'],
  ['.jpeg', 'image/jpeg'],
]);

function serve({ outputDirectory, host, port }) {
  const entrypoint = path.join(outputDirectory, 'index.html');
  if (!fs.existsSync(entrypoint)) {
    fail(`Gallery entrypoint does not exist: '${entrypoint}'. Run the build command first.`);
  }
  const root = fs.realpathSync(outputDirectory);
  const server = http.createServer((request, response) => {
    if (!['GET', 'HEAD'].includes(request.method ?? 'GET')) {
      response.writeHead(405, { Allow: 'GET, HEAD' }).end();
      return;
    }
    let pathname;
    try {
      pathname = decodeURIComponent(new URL(request.url ?? '/', 'http://localhost').pathname);
    } catch {
      response.writeHead(400).end('Bad request');
      return;
    }
    const relative = pathname === '/' ? 'index.html' : pathname.replace(/^\/+/, '');
    const candidate = path.resolve(root, relative);
    if (candidate !== root && !candidate.startsWith(`${root}${path.sep}`)) {
      response.writeHead(403).end('Forbidden');
      return;
    }
    let filePath = candidate;
    try {
      if (fs.statSync(filePath).isDirectory()) filePath = path.join(filePath, 'index.html');
      const body = fs.readFileSync(filePath);
      response.writeHead(200, {
        'Content-Type': MIME_TYPES.get(path.extname(filePath).toLowerCase()) ?? 'application/octet-stream',
        'Cache-Control': 'no-cache',
      });
      if (request.method === 'HEAD') response.end();
      else response.end(body);
    } catch {
      response.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' }).end('Not found');
    }
  });
  server.on('error', (error) => fail(`Could not start gallery server: ${error.message}`));
  server.listen(port, host, () => {
    process.stdout.write(
      `${JSON.stringify(
        {
          status: 'serving',
          directory: root,
          url: `http://${host}:${port}/`,
        },
        null,
        2,
      )}\n`,
    );
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const options = parseArgs(process.argv.slice(2));
  if (options.command === 'build') build(options);
  else serve(options);
}
