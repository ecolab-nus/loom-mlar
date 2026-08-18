import assert from 'node:assert/strict';
import test from 'node:test';

import { planDiagrams, validateReferences } from '../bin/mlar-archify.mjs';
import { buildGalleryCatalog, renderGalleryHtml } from '../lib/gallery.mjs';

const concrete = (value) => ({ text: String(value), constant: value });

function bank(name, capacity = concrete(1024), blockSize = null) {
  return {
    kind: 'bank',
    name,
    capacity,
    block_size: blockSize,
    total_size_bytes: capacity.constant,
  };
}

function memory({ id, scope, name, region = bank(name), dimensions = [] }) {
  return {
    kind: 'memory',
    id,
    scope,
    name,
    dimensions,
    region,
    total_size_bytes: region.total_size_bytes,
  };
}

function actor(kind, id, scope, name) {
  return {
    kind,
    id,
    scope,
    name,
    effect: kind === 'data_mover' ? 'preserve' : 'transform',
    functions: [kind === 'data_mover' ? 'copy' : 'add'],
  };
}

function relationship(id, kind, source, target) {
  return { id, kind, source, target, label: kind };
}

const document = {
  schema_version: 'mlar.visualization.v1',
  architecture: { id: 'architecture-demo', name: 'demo', root_scope: 'scope-system' },
  scopes: [
    { id: 'scope-system', name: 'system', dimensions: [], replication_factor: 1 },
    {
      id: 'scope-core',
      name: 'core',
      parent_scope: 'scope-system',
      dimensions: [{ name: 'tile', size: concrete(4) }],
      replication_factor: 4,
    },
  ],
  components: [
    memory({ id: 'memory-dram', scope: 'scope-system', name: 'DRAM' }),
    memory({
      id: 'memory-l1',
      scope: 'scope-core',
      name: 'L1',
      dimensions: [{ name: 'tile', size: concrete(4) }, { name: 'bank', size: concrete(8) }],
      region: {
        kind: 'array',
        name: 'L1',
        dimensions: [{ name: 'bank', size: concrete(8) }],
        element: bank('L1 bank', concrete(2048), concrete(64)),
        total_size_bytes: 16384,
      },
    }),
    memory({
      id: 'memory-scratch',
      scope: 'scope-core',
      name: 'Scratch',
      region: bank('Scratch', { text: 'SCRATCH_SIZE', constant: null }),
    }),
    memory({ id: 'memory-shared-system', scope: 'scope-system', name: 'Shared' }),
    memory({ id: 'memory-shared-core', scope: 'scope-core', name: 'Shared' }),
    actor('processor', 'processor-lane', 'scope-core', 'lane'),
    actor('data_mover', 'processor-copy', 'scope-system', 'copy'),
    actor('processor', 'processor-orphan', 'scope-system', 'orphan'),
    {
      kind: 'resource',
      id: 'resource-bus',
      scope: 'scope-system',
      name: 'bus',
      resource_kind: 'exclusive',
      capacity: null,
    },
    {
      kind: 'network',
      id: 'network-noc',
      scope: 'scope-system',
      name: 'noc',
      network_kind: 'mesh',
      dimensions: [],
      bandwidth: { text: '32', constant: 32 },
      latency: null,
      links: [],
    },
  ],
  relationships: [
    relationship('relationship-lane-read', 'read', 'memory-l1', 'processor-lane'),
    relationship('relationship-lane-write', 'write', 'processor-lane', 'memory-l1'),
    relationship('relationship-copy-read', 'read', 'memory-dram', 'processor-copy'),
    relationship('relationship-copy-write', 'write', 'processor-copy', 'memory-l1'),
    relationship('relationship-copy-requires', 'requires', 'processor-copy', 'resource-bus'),
    relationship('relationship-noc-attachment', 'network_attachment', 'network-noc', 'memory-l1'),
  ],
};

function planned(source = document) {
  assert.doesNotThrow(() => validateReferences(source));
  return planDiagrams(source);
}

test('US1 plans root-first bounded memory hierarchy with stable structural detail', () => {
  const diagrams = planned();
  const hierarchy = diagrams.filter((diagram) => diagram.section === 'memory_hierarchy');
  assert.equal(hierarchy.length, 1);
  assert.equal(diagrams.filter((diagram) => diagram.section === 'memory_access').length, 0);
  assert.ok(diagrams.every((diagram) => diagram.spec.components.length <= 12));
  assert.ok(diagrams.every((diagram) => diagram.spec.meta.quality_profile === 'showcase'));

  const overview = hierarchy[0];
  assert.equal(overview.id, 'memory-hierarchy-1');
  assert.deepEqual(
    overview.memoryIds,
    ['memory-dram', 'memory-l1', 'memory-scratch', 'memory-shared-core', 'memory-shared-system'],
  );
  assert.deepEqual(overview.scopeIds, ['scope-core', 'scope-system']);
  assert.ok(overview.spec.boundaries.some(
    (boundary) => boundary.label === 'system / core · 4 instances',
  ));
  assert.equal(
    overview.spec.components.find((component) => component.id === 'memory-scratch').tag,
    'unconnected',
  );
  assert.match(
    overview.spec.components.find((component) => component.id === 'memory-scratch').sublabel,
    /SCRATCH_SIZE/,
  );
  assert.ok(overview.spec.components.some((component) => component.id === 'memory-shared-core'));
  assert.ok(overview.spec.components.some((component) => component.id === 'memory-shared-system'));
  assert.ok(overview.componentIds.includes('processor-lane'));
  assert.ok(overview.componentIds.includes('processor-copy'));
  assert.ok(overview.derivedComponentIds.includes('memory-l1-layer-1'));
  assert.ok(!overview.componentIds.includes('memory-l1-layer-1'));
  assert.ok(overview.derivedRelationshipIds.includes('memory-l1-layer-1-contains'));
  assert.deepEqual(overview.relationshipIds.sort(), [
    'relationship-copy-read',
    'relationship-copy-write',
    'relationship-lane-read',
    'relationship-lane-write',
  ]);
  assert.match(
    overview.spec.components.find((component) => component.id === 'memory-l1-layer-1').sublabel,
    /block 64 B/,
  );
  assert.ok(!overview.spec.connections.some((connection) =>
    new Set([connection.from, connection.to]).size === 2
      && [connection.from, connection.to].every((id) =>
        ['memory-dram', 'memory-l1'].includes(id)),
  ), 'scope hierarchy must not imply DRAM/L1 access');
});

test('US1 partitions wide memory hierarchies without expanding instances', () => {
  const wide = structuredClone(document);
  for (let index = 0; index < 12; index += 1) {
    wide.components.push(memory({
      id: `memory-extra-${index}`,
      scope: 'scope-core',
      name: `Extra ${index}`,
    }));
  }
  const diagrams = planned(wide);
  const overviews = diagrams.filter(
    (diagram) => diagram.section === 'memory_hierarchy' && diagram.id.startsWith('memory-hierarchy-'),
  );
  assert.ok(overviews.length > 1);
  assert.ok(overviews.every((diagram) => diagram.spec.components.length <= 12));
  assert.equal(new Set(overviews.flatMap((diagram) => diagram.memoryIds)).size, 17);
  assert.ok(overviews.every((diagram) => !diagram.spec.components.some(
    (component) => component.label.includes('instance 1'),
  )));
});

test('US1 gallery defaults to hierarchy and indexes canonical memory context', () => {
  const diagrams = planned();
  const catalog = buildGalleryCatalog(document, diagrams);
  assert.equal(catalog.schema_version, 'mlar.archify-gallery.v1');
  assert.equal(catalog.language, 'en');
  assert.equal(catalog.default_diagram_id, 'memory-hierarchy-1');
  assert.equal(catalog.sections[0].id, 'memory_hierarchy');
  const hierarchy = catalog.diagrams.find((diagram) => diagram.id === 'memory-hierarchy-1');
  assert.deepEqual(hierarchy.memory_names.sort(), ['DRAM', 'L1', 'Scratch', 'Shared', 'Shared'].sort());
  assert.ok(hierarchy.scope_ids.includes('scope-core'));
});

test('US2 combines read and write access while preserving actor roles and mover routes', () => {
  const diagrams = planned();
  const l1Access = diagrams.find(
    (diagram) => diagram.section === 'memory_hierarchy'
      && diagram.memoryIds.includes('memory-l1')
      && diagram.componentIds.includes('processor-lane'),
  );
  assert.ok(l1Access);
  assert.equal(
    l1Access.componentIds.filter((id) => id === 'processor-lane').length,
    1,
  );
  assert.ok(l1Access.relationshipIds.includes('relationship-lane-read'));
  assert.ok(l1Access.relationshipIds.includes('relationship-lane-write'));
  assert.equal(
    l1Access.spec.components.find((component) => component.id === 'processor-lane').type,
    'backend',
  );

  const moverView = diagrams.find(
    (diagram) => diagram.section === 'memory_hierarchy'
      && diagram.componentIds.includes('processor-copy'),
  );
  assert.ok(moverView);
  assert.ok(moverView.componentIds.includes('memory-dram'));
  assert.ok(moverView.componentIds.includes('memory-l1'));
  assert.equal(
    moverView.spec.components.find((component) => component.id === 'processor-copy').type,
    'messagebus',
  );
  assert.deepEqual(
    moverView.spec.connections
      .filter((connection) => connection.id.startsWith('relationship-copy-'))
      .map(({ id, from, to, label }) => ({ id, from, to, label }))
      .sort((left, right) => left.id.localeCompare(right.id)),
    [
      {
        id: 'relationship-copy-read',
        from: 'memory-dram',
        to: 'processor-copy',
        label: 'read',
      },
      {
        id: 'relationship-copy-write',
        from: 'processor-copy',
        to: 'memory-l1',
        label: 'write',
      },
    ],
  );
  assert.ok(!moverView.spec.connections.some((connection) =>
    connection.from === 'memory-dram' && connection.to === 'memory-l1'),
  'scope or structural hierarchy must not infer read/write access');
});

test('US2 keeps same-memory mover meanings without duplicating its identity', () => {
  const sameMemory = structuredClone(document);
  sameMemory.components.push(actor('data_mover', 'processor-loop', 'scope-system', 'loop'));
  sameMemory.relationships.push(
    relationship('relationship-loop-read', 'read', 'memory-dram', 'processor-loop'),
    relationship('relationship-loop-write', 'write', 'processor-loop', 'memory-dram'),
  );
  const view = planned(sameMemory).find(
    (diagram) => diagram.section === 'memory_hierarchy'
      && diagram.componentIds.includes('processor-loop'),
  );
  assert.ok(view);
  assert.equal(view.componentIds.filter((id) => id === 'processor-loop').length, 1);
  assert.equal(view.componentIds.filter((id) => id === 'memory-dram').length, 1);
  assert.deepEqual(
    view.relationshipIds.filter((id) => id.startsWith('relationship-loop-')).sort(),
    ['relationship-loop-read', 'relationship-loop-write'],
  );
});

test('US2 deterministically partitions more than twelve direct neighbors', () => {
  const large = structuredClone(document);
  for (let index = 0; index < 13; index += 1) {
    const id = `processor-scratch-${index}`;
    large.components.push(actor('processor', id, 'scope-core', `scratch ${index}`));
    large.relationships.push(
      relationship(`relationship-scratch-${index}`, 'read', 'memory-scratch', id),
    );
  }
  const pages = planned(large).filter(
    (diagram) => diagram.section === 'memory_access'
      && diagram.memoryIds.includes('memory-scratch'),
  );
  assert.equal(pages.length, 2);
  assert.ok(pages.every((diagram) => diagram.spec.components.length <= 12));
  assert.equal(
    new Set(pages.flatMap((diagram) => diagram.componentIds)
      .filter((id) => id.startsWith('processor-scratch-'))).size,
    13,
  );
  assert.equal(
    new Set(pages.flatMap((diagram) => diagram.relationshipIds)
      .filter((id) => id.startsWith('relationship-scratch-'))).size,
    13,
  );
  assert.deepEqual(
    pages.map((page) => page.id),
    ['memory-access-memory-scratch-1', 'memory-access-memory-scratch-2'],
  );
});

test('US2 gallery indexes all canonical memories in the unified route view', () => {
  const catalog = buildGalleryCatalog(document, planned());
  const moverEntry = catalog.diagrams.find(
    (diagram) => diagram.section === 'memory_hierarchy'
      && diagram.memory_ids.includes('memory-dram')
      && diagram.memory_ids.includes('memory-l1'),
  );
  assert.ok(moverEntry);
  assert.deepEqual(
    moverEntry.memory_names.sort(),
    ['DRAM', 'L1', 'Scratch', 'Shared', 'Shared'].sort(),
  );
});

test('US3 preserves routes, supporting relationships, and complete canonical coverage', () => {
  const diagrams = planned();
  const supporting = diagrams.filter((diagram) => diagram.section === 'supporting_context');
  assert.ok(supporting.some((diagram) => diagram.relationshipIds.includes(
    'relationship-copy-requires',
  )));
  assert.ok(supporting.some((diagram) => diagram.relationshipIds.includes(
    'relationship-noc-attachment',
  )));
  assert.ok(supporting.some((diagram) => diagram.componentIds.includes('processor-orphan')));

  assert.deepEqual(
    [...new Set(diagrams.flatMap((diagram) => diagram.componentIds))].sort(),
    document.components.map((component) => component.id).sort(),
  );
  assert.deepEqual(
    [...new Set(diagrams.flatMap((diagram) => diagram.relationshipIds))].sort(),
    document.relationships.map((relationship) => relationship.id).sort(),
  );
  assert.deepEqual(
    [...new Set(diagrams.flatMap((diagram) => diagram.scopeIds))].sort(),
    document.scopes.map((scope) => scope.id).sort(),
  );
  assert.ok(diagrams.every((diagram) => diagram.spec.components.length <= 12));

  for (const memoryId of ['memory-dram', 'memory-l1']) {
    const route = diagrams.find(
      (diagram) => diagram.section === 'memory_hierarchy'
        && diagram.memoryIds.includes(memoryId)
        && diagram.componentIds.includes('processor-copy'),
    );
    assert.ok(route);
    assert.ok(route.componentIds.includes('memory-dram'));
    assert.ok(route.componentIds.includes('memory-l1'));
  }
});

test('US3 partitions dense supporting context by canonical source', () => {
  const dense = structuredClone(document);
  for (let index = 0; index < 13; index += 1) {
    const resourceId = `resource-copy-${index}`;
    dense.components.push({
      kind: 'resource',
      id: resourceId,
      scope: 'scope-system',
      name: `copy resource ${index}`,
      resource_kind: 'exclusive',
      capacity: null,
    });
    dense.relationships.push(
      relationship(
        `relationship-copy-resource-${index}`,
        'requires',
        'processor-copy',
        resourceId,
      ),
    );
  }
  const pages = planned(dense).filter(
    (diagram) => diagram.section === 'supporting_context'
      && diagram.componentIds.includes('processor-copy')
      && diagram.relationshipIds.some((id) => id.startsWith('relationship-copy-resource-')),
  );
  assert.equal(pages.length, 2);
  assert.ok(pages.every((diagram) => diagram.spec.components.length <= 12));
  assert.equal(
    new Set(pages.flatMap((diagram) => diagram.relationshipIds)
      .filter((id) => id.startsWith('relationship-copy-resource-'))).size,
    13,
  );
});

test('US3 planning and catalog output are deterministic and memory-first', () => {
  const first = planned();
  const second = planned();
  assert.deepEqual(second, first);
  const catalog = buildGalleryCatalog(document, first);
  assert.deepEqual(
    catalog.sections.map((section) => section.id),
    ['memory_hierarchy', 'memory_access', 'supporting_context', 'other'],
  );
  assert.equal(catalog.default_diagram_id, 'memory-hierarchy-1');
});

test('US3 gallery remains an English Archify shell without a second renderer', () => {
  const html = renderGalleryHtml(document, planned());
  assert.match(html, /<html lang="en">/);
  assert.match(html, /<iframe id="diagramFrame"/);
  assert.match(html, /window\.addEventListener\('hashchange'/);
  assert.match(html, /event\.key === 'ArrowLeft'/);
  assert.match(html, /window\.open\(diagram\.html/);
  assert.doesNotMatch(html, /<svg\b/i);
});

test('reference validation rejects disconnected scopes and unknown endpoints', () => {
  const disconnected = structuredClone(document);
  disconnected.scopes.push({
    id: 'scope-orphan',
    name: 'orphan',
    dimensions: [],
    replication_factor: 1,
  });
  assert.throws(() => validateReferences(disconnected), /not connected to the root scope/);

  const unknownEndpoint = structuredClone(document);
  unknownEndpoint.relationships[0].target = 'processor-missing';
  assert.throws(() => validateReferences(unknownEndpoint), /unknown endpoint/);
});
