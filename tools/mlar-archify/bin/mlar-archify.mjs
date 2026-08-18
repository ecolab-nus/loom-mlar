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

function scopePath(scopeId, scopesById) {
  const result = [];
  const visited = new Set();
  let cursor = scopesById.get(scopeId);
  while (cursor && !visited.has(cursor.id)) {
    visited.add(cursor.id);
    result.unshift(cursor);
    cursor = cursor.parent_scope ? scopesById.get(cursor.parent_scope) : null;
  }
  return result;
}

function scopeContextLabel(scopeId, scopesById) {
  const path = scopePath(scopeId, scopesById);
  const replication = path.at(-1)?.replication_factor ?? 1;
  const suffix = replication > 1 ? ` · ${replication} instances` : '';
  return `${path.map((scope) => scope.name).join(' / ')}${suffix}`;
}

function regionSummary(region) {
  if (region.kind === 'array') {
    return [
      'array',
      dimensionsLabel(region.dimensions),
      formatBytes(region.total_size_bytes),
    ].filter(Boolean).join(' · ');
  }
  return [
    region.capacity?.text ? `capacity ${region.capacity.text} B` : null,
    region.block_size?.text ? `block ${region.block_size.text} B` : null,
    formatBytes(region.total_size_bytes),
  ].filter(Boolean).join(' · ');
}

function derivedLayer(memory, region, depth, parentId) {
  const id = `${memory.id}-layer-${depth}`;
  return {
    component: {
      kind: 'memory_layer',
      id,
      name: region.name || (region.kind === 'array' ? `Array level ${depth}` : 'Bank'),
      layer_kind: region.kind,
      dimensions: region.dimensions ?? [],
      capacity: region.capacity ?? null,
      block_size: region.block_size ?? null,
      total_size_bytes: region.total_size_bytes ?? null,
      derived: true,
      canonical_memory_id: memory.id,
      scope: memory.scope,
    },
    relationship: {
      id: `${id}-contains`,
      kind: 'contains',
      source: parentId,
      target: id,
      label: 'contains',
      derived: true,
    },
  };
}

function projectMemoryLayers(memory) {
  const components = [];
  const relationships = [];
  let region = memory.region;
  let parentId = memory.id;
  let depth = 1;
  while (region?.kind === 'array') {
    region = region.element;
    const layer = derivedLayer(memory, region, depth, parentId);
    components.push(layer.component);
    relationships.push(layer.relationship);
    parentId = layer.component.id;
    depth += 1;
  }
  return { components, relationships };
}

function buildProjection(document) {
  const scopesById = new Map(document.scopes.map((scope) => [scope.id, scope]));
  const componentsById = new Map(document.components.map((component) => [component.id, component]));
  const memories = document.components
    .filter((component) => component.kind === 'memory')
    .sort((left, right) => left.id.localeCompare(right.id));
  const memoryIds = new Set(memories.map((memory) => memory.id));
  const actorIds = new Set(document.components
    .filter((component) => component.kind === 'processor' || component.kind === 'data_mover')
    .map((actor) => actor.id));
  const accessByActor = new Map();
  const actorIdsByMemory = new Map(memories.map((memory) => [memory.id, new Set()]));

  for (const relationship of document.relationships) {
    if (relationship.kind !== 'read' && relationship.kind !== 'write') continue;
    const memoryId = relationship.kind === 'read' ? relationship.source : relationship.target;
    const actorId = relationship.kind === 'read' ? relationship.target : relationship.source;
    if (!memoryIds.has(memoryId) || !actorIds.has(actorId)) continue;
    if (!accessByActor.has(actorId)) {
      accessByActor.set(actorId, {
        actor: componentsById.get(actorId),
        endpointIds: new Set(),
        relationships: [],
      });
    }
    const unit = accessByActor.get(actorId);
    unit.endpointIds.add(memoryId);
    unit.relationships.push(relationship);
    actorIdsByMemory.get(memoryId).add(actorId);
  }

  for (const unit of accessByActor.values()) {
    unit.endpointIds = [...unit.endpointIds].sort();
    unit.relationships.sort((left, right) => left.id.localeCompare(right.id));
  }

  const layersByMemory = new Map(
    memories.map((memory) => [memory.id, projectMemoryLayers(memory)]),
  );
  return {
    scopesById,
    componentsById,
    memories,
    accessByActor,
    actorIdsByMemory,
    layersByMemory,
  };
}

function archifyComponent(component, index, columns) {
  const base = {
    id: component.id,
    label: component.name,
    row: component.row ?? Math.floor(index / columns),
    col: component.col ?? index % columns,
    size: [320, 62],
  };
  switch (component.kind) {
    case 'memory': {
      const details = component.region.kind === 'array'
        ? ['array', dimensionsLabel(component.dimensions), formatBytes(component.total_size_bytes)]
        : ['bank', dimensionsLabel(component.dimensions), regionSummary(component.region)];
      const sublabel = details
        .filter(Boolean)
        .join(' · ');
      return {
        ...base,
        type: 'database',
        sublabel: sublabel || 'memory',
        tag: component.connection_count === 0 ? 'unconnected' : undefined,
      };
    }
    case 'memory_layer': {
      const details = component.layer_kind === 'array'
        ? [dimensionsLabel(component.dimensions), formatBytes(component.total_size_bytes)]
        : [
            component.capacity?.text ? `capacity ${component.capacity.text} B` : null,
            component.block_size?.text ? `block ${component.block_size.text} B` : null,
            formatBytes(component.total_size_bytes),
          ];
      return {
        ...base,
        type: 'database',
        sublabel: details.filter(Boolean).join(' · ') || component.layer_kind,
        tag: component.layer_kind,
      };
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
    requires: 'requires',
    network_attachment: 'attaches',
    contains: 'contains',
  };
  const connection = {
    id: relationship.id,
    from: relationship.source,
    to: relationship.target,
  };
  const label = labels[relationship.kind];
  if (label) connection.label = label;
  if (relationship.kind === 'requires') connection.variant = 'dashed';
  if (relationship.kind === 'requires') connection.labelDy = 24;
  if (relationship.kind === 'read' || relationship.kind === 'write') {
    connection.variant = 'emphasis';
  }
  if (relationship.kind === 'contains') {
    connection.variant = 'dashed';
    connection.labelDy = 24;
  }
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
  section = 'supporting_context',
  primaryScopeId = null,
  components,
  relationships = [],
  boundaries = [],
  sourceScopeIds = [],
  memoryIds = [],
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
  const explicitColumns = sortedComponents.reduce(
    (maximum, component) => Math.max(maximum, (component.col ?? -1) + 1),
    0,
  );
  const columns = Math.min(
    12,
    Math.max(1, explicitColumns || Math.min(4, Math.ceil(Math.sqrt(sortedComponents.length)))),
  );
  const selectedBoundaries = boundaries
    .map((boundary) => ({
      ...boundary,
      wraps: boundary.wraps.filter((componentId) => componentIds.has(componentId)),
    }))
    .filter((boundary) => boundary.wraps.length > 0);
  const presentationComponents = sortedComponents.map((component, index) =>
    archifyComponent(component, index, columns),
  );
  const layout = {
    mode: 'grid',
    origin: [40, 70],
    cols: columns,
    gapX: 86,
    gapY: 76,
    cellW: 320,
    cellH: 76,
  };
  const spec = {
    schema_version: 1,
    diagram_type: 'architecture',
    meta: {
      title,
      quality_profile: 'showcase',
    },
    layout,
    components: presentationComponents,
    connections: selectedRelationships.map(archifyConnection),
  };
  if (selectedBoundaries.length > 0) spec.boundaries = selectedBoundaries;
  return {
    id,
    title,
    type: 'architecture',
    section,
    primaryScopeId,
    spec,
    componentIds: sortedComponents
      .filter((component) => component.kind !== 'scope' && !component.derived)
      .map((component) => component.id),
    derivedComponentIds: sortedComponents
      .filter((component) => component.derived)
      .map((component) => component.id),
    scopeIds: [...new Set([
      ...sourceScopeIds,
      ...sortedComponents
        .filter((component) => component.kind === 'scope')
        .map((component) => component.id),
    ])].sort(),
    relationshipIds: selectedRelationships
      .filter((relationship) => !relationship.derived)
      .map((relationship) => relationship.id),
    derivedRelationshipIds: selectedRelationships
      .filter((relationship) => relationship.derived)
      .map((relationship) => relationship.id),
    memoryIds: [...new Set(memoryIds)].sort(),
  };
}

function chunks(values, size) {
  const result = [];
  for (let index = 0; index < values.length; index += size) {
    result.push(values.slice(index, index + size));
  }
  return result;
}

function scopeIdsForComponents(components, scopesById) {
  const result = new Set();
  for (const component of components) {
    if (!component.scope) continue;
    for (const scope of scopePath(component.scope, scopesById)) result.add(scope.id);
  }
  return [...result].sort();
}

function unifiedMemoryDiagram(document, projection) {
  if (projection.memories.length === 0) return null;

  const memories = projection.memories.map((memory) => ({
    ...memory,
    connection_count: projection.actorIdsByMemory.get(memory.id)?.size ?? 0,
  }));
  const actorUnits = [...projection.accessByActor.values()]
    .sort((left, right) => left.actor.id.localeCompare(right.actor.id));
  const derivedLayers = memories.flatMap(
    (memory) => projection.layersByMemory.get(memory.id).components,
  );
  if (memories.length + derivedLayers.length + actorUnits.length > MAX_COMPONENTS) {
    return null;
  }

  const maximumLayerCount = Math.max(
    0,
    ...memories.map((memory) => projection.layersByMemory.get(memory.id).components.length),
  );
  const memoryRowStride = maximumLayerCount + 2;
  const actorMiddleRow = Math.max(0, Math.floor((actorUnits.length - 1) / 2));
  const memoryPositions = new Map();
  const memoryCountByDepth = new Map();
  for (const memory of memories) {
    const depth = scopePath(memory.scope, projection.scopesById).length - 1;
    const indexAtDepth = memoryCountByDepth.get(depth) ?? 0;
    memoryCountByDepth.set(depth, indexAtDepth + 1);
    memoryPositions.set(memory.id, {
      row: actorMiddleRow + indexAtDepth * memoryRowStride,
      col: depth * 2,
    });
  }
  const components = [];
  const relationships = [];
  for (const memory of memories) {
    const position = memoryPositions.get(memory.id);
    components.push({ ...memory, ...position });
    const layers = projection.layersByMemory.get(memory.id);
    layers.components.forEach((layer, index) => {
      components.push({ ...layer, row: position.row + index + 1, col: position.col });
    });
    relationships.push(...layers.relationships);
  }
  actorUnits.forEach((unit, index) => {
    const endpointColumns = unit.endpointIds
      .map((memoryId) => memoryPositions.get(memoryId).col);
    const shallowestColumn = Math.min(...endpointColumns);
    const deepestColumn = Math.max(...endpointColumns);
    const col = shallowestColumn === deepestColumn
      ? (shallowestColumn === 0 ? 1 : shallowestColumn - 1)
      : shallowestColumn + 1;
    components.push({
      ...unit.actor,
      row: index,
      col,
    });
    relationships.push(...unit.relationships);
  });

  const memoryAndLayerIdsByScope = new Map();
  for (const component of components) {
    if (component.kind !== 'memory' && component.kind !== 'memory_layer') continue;
    if (!memoryAndLayerIdsByScope.has(component.scope)) {
      memoryAndLayerIdsByScope.set(component.scope, []);
    }
    memoryAndLayerIdsByScope.get(component.scope).push(component.id);
  }
  const boundaries = [...memoryAndLayerIdsByScope.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([scopeId, wraps]) => ({
      kind: 'region',
      label: scopeContextLabel(scopeId, projection.scopesById),
      wraps,
    }));

  return createDiagram({
    id: 'memory-hierarchy-1',
    title: 'Memory hierarchy and access',
    section: 'memory_hierarchy',
    primaryScopeId: document.architecture.root_scope,
    components,
    relationships,
    boundaries,
    sourceScopeIds: scopeIdsForComponents(components, projection.scopesById),
    memoryIds: memories.map((memory) => memory.id),
  });
}

function hierarchyDiagrams(document, projection) {
  const diagrams = [];
  const memoryPresentations = projection.memories.map((memory) => ({
    ...memory,
    connection_count: projection.actorIdsByMemory.get(memory.id)?.size ?? 0,
  }));
  const overviewChunks = memoryPresentations.length === 0
    ? [[]]
    : chunks(memoryPresentations, MAX_COMPONENTS);
  for (const [chunkIndex, memoryChunk] of overviewChunks.entries()) {
    const components = memoryChunk.length > 0
      ? memoryChunk
      : [{
          kind: 'scope',
          id: document.architecture.root_scope,
          name: document.architecture.name,
          dimensions: [],
          replication_factor: 1,
        }];
    const scopeIds = new Set();
    const boundaries = [];
    const byScope = new Map();
    for (const memory of memoryChunk) {
      if (!byScope.has(memory.scope)) byScope.set(memory.scope, []);
      byScope.get(memory.scope).push(memory.id);
      for (const scope of scopePath(memory.scope, projection.scopesById)) scopeIds.add(scope.id);
    }
    for (const [scopeId, wraps] of [...byScope.entries()].sort()) {
      boundaries.push({
        kind: 'region',
        label: scopeContextLabel(scopeId, projection.scopesById),
        wraps,
      });
    }
    if (memoryChunk.length === 0) scopeIds.add(document.architecture.root_scope);
    const suffix = overviewChunks.length > 1 ? ` · ${chunkIndex + 1}` : '';
    const diagram = createDiagram({
      id: diagramId(['memory-hierarchy', String(chunkIndex + 1)]),
      title: `Memory hierarchy${suffix}`,
      section: 'memory_hierarchy',
      primaryScopeId: document.architecture.root_scope,
      components,
      boundaries,
      sourceScopeIds: [...scopeIds],
      memoryIds: memoryChunk.map((memory) => memory.id),
    });
    if (diagram) diagrams.push(diagram);
  }

  for (const memory of memoryPresentations) {
    const layers = projection.layersByMemory.get(memory.id);
    if (layers.components.length === 0) continue;
    let cursor = 0;
    let previousLayer = null;
    let page = 1;
    while (cursor < layers.components.length) {
      const pageCapacity = previousLayer ? MAX_COMPONENTS - 2 : MAX_COMPONENTS - 1;
      const pageLayers = layers.components.slice(cursor, cursor + pageCapacity);
      const components = [memory, ...(previousLayer ? [previousLayer] : []), ...pageLayers]
        .map((component, index) => ({ ...component, row: index, col: 0 }));
      const ids = new Set(components.map((component) => component.id));
      const relationships = layers.relationships.filter(
        (relationship) => ids.has(relationship.source) && ids.has(relationship.target),
      );
      const pathNames = scopePath(memory.scope, projection.scopesById).map((scope) => scope.name);
      const pageSuffix = layers.components.length > pageCapacity ? ` · ${page}` : '';
      const diagram = createDiagram({
        id: diagramId(['memory-structure', memory.id, String(page)]),
        title: `Memory structure · ${[...pathNames, memory.name].join(' / ')}${pageSuffix}`,
        section: 'memory_hierarchy',
        primaryScopeId: memory.scope,
        components,
        relationships,
        boundaries: [{
          kind: 'region',
          label: scopeContextLabel(memory.scope, projection.scopesById),
          wraps: components.map((component) => component.id),
        }],
        sourceScopeIds: scopePath(memory.scope, projection.scopesById).map((scope) => scope.id),
        memoryIds: [memory.id],
      });
      if (diagram) diagrams.push(diagram);
      cursor += pageLayers.length;
      previousLayer = pageLayers.at(-1);
      page += 1;
    }
  }
  return diagrams;
}

function packActorUnits(anchorId, actorIds, projection) {
  const result = [];
  let units = [];
  let ids = new Set([anchorId]);
  const flush = () => {
    if (units.length > 0) result.push(units);
    units = [];
    ids = new Set([anchorId]);
  };
  for (const actorId of actorIds) {
    const unit = projection.accessByActor.get(actorId);
    const unitIds = new Set([actorId, ...unit.endpointIds]);
    const candidate = new Set([...ids, ...unitIds]);
    if (candidate.size > MAX_COMPONENTS && units.length > 0) flush();
    units.push(unit);
    ids = new Set([...ids, ...unitIds]);
  }
  flush();
  return result;
}

function accessDiagrams(projection) {
  const diagrams = [];
  for (const memory of projection.memories) {
    const actorIds = [...(projection.actorIdsByMemory.get(memory.id) ?? [])].sort();
    if (actorIds.length === 0) continue;
    const unitChunks = packActorUnits(memory.id, actorIds, projection);
    for (const [chunkIndex, units] of unitChunks.entries()) {
      const ids = new Set([memory.id]);
      const relationshipById = new Map();
      for (const unit of units) {
        ids.add(unit.actor.id);
        for (const endpointId of unit.endpointIds) ids.add(endpointId);
        for (const relationship of unit.relationships) {
          relationshipById.set(relationship.id, relationship);
        }
      }
      const components = [...ids].map((id) => ({ ...projection.componentsById.get(id) }));
      const positions = new Map([[
        memory.id,
        { row: 0, col: Math.floor((units.length - 1) / 2) },
      ]]);
      units.forEach((unit, index) => {
        positions.set(unit.actor.id, { row: 1, col: index });
        for (const relationship of unit.relationships) {
          const endpointId = relationship.kind === 'read'
            ? relationship.source
            : relationship.target;
          if (endpointId === memory.id || positions.has(endpointId)) continue;
          positions.set(endpointId, {
            row: 2,
            col: index,
          });
        }
      });
      for (const component of components) Object.assign(component, positions.get(component.id));
      const pathNames = scopePath(memory.scope, projection.scopesById).map((scope) => scope.name);
      const suffix = unitChunks.length > 1 ? ` · ${chunkIndex + 1}` : '';
      const diagram = createDiagram({
        id: diagramId(['memory-access', memory.id, String(chunkIndex + 1)]),
        title: `Memory access · ${[...pathNames, memory.name].join(' / ')}${suffix}`,
        section: 'memory_access',
        primaryScopeId: memory.scope,
        components,
        relationships: [...relationshipById.values()],
        sourceScopeIds: scopeIdsForComponents(components, projection.scopesById),
        memoryIds: components
          .filter((component) => component.kind === 'memory')
          .map((component) => component.id),
      });
      if (diagram) diagrams.push(diagram);
    }
  }
  return diagrams;
}

function supportingDiagrams(document, projection, existingDiagrams) {
  const diagrams = [];
  const coveredRelationships = new Set(
    existingDiagrams.flatMap((diagram) => diagram.relationshipIds),
  );
  const remainingRelationships = document.relationships
    .filter((relationship) => !coveredRelationships.has(relationship.id))
    .sort((left, right) => left.id.localeCompare(right.id));
  const relationshipsBySource = new Map();
  for (const relationship of remainingRelationships) {
    if (!relationshipsBySource.has(relationship.source)) {
      relationshipsBySource.set(relationship.source, []);
    }
    relationshipsBySource.get(relationship.source).push(relationship);
  }
  for (const [sourceId, sourceRelationships] of [...relationshipsBySource.entries()].sort()) {
    for (const [chunkIndex, relationshipChunk] of chunks(
      sourceRelationships,
      MAX_COMPONENTS - 1,
    ).entries()) {
      const targetIds = [...new Set(
        relationshipChunk.map((relationship) => relationship.target),
      )].sort();
      const source = { ...projection.componentsById.get(sourceId) };
      source.row = 0;
      source.col = Math.floor((targetIds.length - 1) / 2);
      const components = [
        source,
        ...targetIds.map((targetId, index) => ({
          ...projection.componentsById.get(targetId),
          row: 1,
          col: index,
        })),
      ];
      const suffix = sourceRelationships.length > MAX_COMPONENTS - 1
        ? ` · ${chunkIndex + 1}`
        : '';
      const diagram = createDiagram({
        id: diagramId(['supporting-relationships', sourceId, String(chunkIndex + 1)]),
        title: `Supporting context · ${source.name}${suffix}`,
        section: 'supporting_context',
        primaryScopeId: source.scope ?? document.architecture.root_scope,
        components,
        relationships: relationshipChunk,
        sourceScopeIds: scopeIdsForComponents(components, projection.scopesById),
        memoryIds: components
          .filter((component) => component.kind === 'memory')
          .map((component) => component.id),
      });
      if (diagram) diagrams.push(diagram);
    }
  }

  const allDiagrams = [...existingDiagrams, ...diagrams];
  const coveredComponents = new Set(allDiagrams.flatMap((diagram) => diagram.componentIds));
  const uncoveredComponents = document.components
    .filter((component) => !coveredComponents.has(component.id))
    .sort((left, right) => left.id.localeCompare(right.id));
  for (const componentChunk of chunks(uncoveredComponents, MAX_COMPONENTS)) {
    const diagram = createDiagram({
      id: diagramId(['supporting-components', String(diagrams.length + 1)]),
      title: `Supporting context · components ${diagrams.length + 1}`,
      section: 'supporting_context',
      primaryScopeId: componentChunk[0]?.scope ?? document.architecture.root_scope,
      components: componentChunk,
      sourceScopeIds: scopeIdsForComponents(componentChunk, projection.scopesById),
      memoryIds: componentChunk
        .filter((component) => component.kind === 'memory')
        .map((component) => component.id),
    });
    if (diagram) diagrams.push(diagram);
  }

  const coveredScopes = new Set(
    [...existingDiagrams, ...diagrams].flatMap((diagram) => diagram.scopeIds),
  );
  const uncoveredScopes = document.scopes
    .filter((scope) => !coveredScopes.has(scope.id))
    .sort((left, right) => left.id.localeCompare(right.id));
  for (const scopeChunk of chunks(uncoveredScopes, MAX_COMPONENTS)) {
    const scopeComponents = scopeChunk.map((scope) => ({
      kind: 'scope',
      id: scope.id,
      name: scope.name,
      dimensions: scope.dimensions,
      replication_factor: scope.replication_factor,
    }));
    const diagram = createDiagram({
      id: diagramId(['supporting-scopes', String(diagrams.length + 1)]),
      title: `Supporting context · scopes ${diagrams.length + 1}`,
      section: 'supporting_context',
      primaryScopeId: scopeChunk[0]?.id ?? document.architecture.root_scope,
      components: scopeComponents,
      sourceScopeIds: scopeChunk.map((scope) => scope.id),
    });
    if (diagram) diagrams.push(diagram);
  }
  return diagrams;
}

export function planDiagrams(document) {
  const projection = buildProjection(document);
  const unified = unifiedMemoryDiagram(document, projection);
  const diagrams = unified ? [unified] : hierarchyDiagrams(document, projection);
  if (!unified) diagrams.push(...accessDiagrams(projection));
  diagrams.push(...supportingDiagrams(document, projection, diagrams));

  const ids = new Set();
  for (const diagram of diagrams) {
    if (ids.has(diagram.id)) fail(`Duplicate diagram id '${diagram.id}'.`);
    ids.add(diagram.id);
  }
  return diagrams;
}

function runArchify(args, { allowSkipped = false } = {}) {
  const result = spawnSync(process.execPath, [archifyBin, ...args], {
    cwd: repoRoot,
    encoding: 'utf8',
  });
  const output = result.stdout.trim();
  let parsed;
  try {
    parsed = JSON.parse(output);
  } catch {
    parsed = { raw: output };
  }
  if (allowSkipped && result.status === 2 && parsed.status === 'skipped') {
    return parsed;
  }
  if (result.status !== 0) {
    fail(`Archify command failed: ${args.join(' ')}`, {
      status: result.status,
      stdout: result.stdout,
      stderr: result.stderr,
    });
  }
  return parsed;
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
      ? runArchify(['visual-check', htmlPath, '--json'], { allowSkipped: true })
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
      derived_component_ids: diagram.derivedComponentIds,
      scope_ids: diagram.scopeIds,
      relationship_ids: diagram.relationshipIds,
      derived_relationship_ids: diagram.derivedRelationshipIds,
      memory_ids: diagram.memoryIds,
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
  const coveredDerivedComponents = new Set(
    manifestDiagrams.flatMap((diagram) => diagram.derived_component_ids),
  );
  const coveredDerivedRelationships = new Set(
    manifestDiagrams.flatMap((diagram) => diagram.derived_relationship_ids),
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
    included_derived_component_ids: [...coveredDerivedComponents].sort(),
    included_derived_relationship_ids: [...coveredDerivedRelationships].sort(),
    policy: {
      maximum_primary_nodes: MAX_COMPONENTS,
      expand_replicated_instances: false,
      create_synthetic_collapsed_nodes: false,
      partition_strategy: 'unified_memory_view_then_bounded_overflow_then_supporting_context',
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
