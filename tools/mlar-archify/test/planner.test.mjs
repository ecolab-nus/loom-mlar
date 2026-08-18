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

test('US1 plans a bounded System View with stable structural detail', () => {
  const diagrams = planned();
  const systemViews = diagrams.filter((diagram) => diagram.section === 'system_view');
  assert.equal(systemViews.length, 1);
  assert.ok(diagrams.every((diagram) => diagram.spec.components.length <= 12));
  assert.ok(diagrams.every((diagram) => diagram.spec.meta.quality_profile === 'showcase'));

  const overview = systemViews[0];
  assert.equal(overview.id, 'system-view-1');
  assert.equal(overview.title, 'System View');
  assert.deepEqual(overview.spec.meta.legend, {
    mode: 'auto',
    entries: {
      backend: { label: 'Processor' },
      database: { label: 'Memory' },
      cloud: { label: 'Resource' },
      messagebus: { label: 'Data Mover' },
      external: { label: 'Network' },
      frontend: { label: 'Architecture Scope' },
    },
  });
  assert.equal(
    overview.spec.meta.subtitle,
    'Arrows show source memory → processor or data mover → destination memory; boundaries show architecture scopes.',
  );
  assert.deepEqual(
    overview.memoryIds,
    ['memory-dram', 'memory-l1', 'memory-scratch', 'memory-shared-core', 'memory-shared-system'],
  );
  assert.deepEqual(overview.scopeIds, ['scope-core', 'scope-system']);
  assert.ok(overview.spec.boundaries.some(
    (boundary) => boundary.label === 'system / core · 4 instances',
  ));
  assert.ok(overview.spec.boundaries.some(
    (boundary) => boundary.wraps.includes('processor-lane'),
  ));
  const rootBoundary = overview.spec.boundaries.find(
    (boundary) => boundary.label === 'system',
  );
  const childBoundary = overview.spec.boundaries.find(
    (boundary) => boundary.label === 'system / core · 4 instances',
  );
  assert.ok(rootBoundary);
  assert.ok(childBoundary);
  assert.ok(childBoundary.wraps.every((componentId) => rootBoundary.wraps.includes(componentId)),
    'a child scope boundary must be nested inside its ancestor boundary');
  assert.deepEqual(
    [...rootBoundary.wraps].sort(),
    overview.spec.components.map((component) => component.id).sort(),
    'the root scope boundary must contain every component shown in System View',
  );
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

test('US1 gives sibling architecture scopes non-overlapping System View row bands', () => {
  const siblings = structuredClone(document);
  siblings.scopes = [
    siblings.scopes[0],
    {
      id: 'scope-a',
      name: 'a',
      parent_scope: 'scope-system',
      dimensions: [],
      replication_factor: 1,
    },
    {
      id: 'scope-b',
      name: 'b',
      parent_scope: 'scope-system',
      dimensions: [],
      replication_factor: 1,
    },
  ];
  const arrayRegion = (name) => ({
    kind: 'array',
    name,
    dimensions: [{ name: 'bank', size: concrete(2) }],
    element: bank(`${name} bank`),
    total_size_bytes: 2048,
  });
  siblings.components = [
    memory({ id: 'memory-a', scope: 'scope-a', name: 'A', region: arrayRegion('A') }),
    memory({ id: 'memory-b', scope: 'scope-b', name: 'B', region: arrayRegion('B') }),
    actor('processor', 'processor-a', 'scope-a', 'processor a'),
    actor('processor', 'processor-b', 'scope-b', 'processor b'),
  ];
  siblings.relationships = [
    relationship('relationship-a-read', 'read', 'memory-a', 'processor-a'),
    relationship('relationship-a-write', 'write', 'processor-a', 'memory-a'),
    relationship('relationship-b-read', 'read', 'memory-b', 'processor-b'),
    relationship('relationship-b-write', 'write', 'processor-b', 'memory-b'),
  ];

  const overview = planned(siblings).find((diagram) => diagram.id === 'system-view-1');
  assert.ok(overview);
  const componentsById = new Map(
    overview.spec.components.map((component) => [component.id, component]),
  );
  const boundaryRange = (label) => {
    const boundary = overview.spec.boundaries.find((candidate) => candidate.label === label);
    assert.ok(boundary);
    const rows = boundary.wraps.map((componentId) => componentsById.get(componentId).row);
    return { minimum: Math.min(...rows), maximum: Math.max(...rows) };
  };
  const a = boundaryRange('system / a');
  const b = boundaryRange('system / b');
  assert.ok(
    a.maximum + 1 < b.minimum || b.maximum + 1 < a.minimum,
    'sibling scope boundaries must have a clear row between them',
  );
});

test('US1 partitions a wide System View without expanding instances', () => {
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
    (diagram) => diagram.section === 'system_view' && diagram.id.startsWith('system-view-'),
  );
  assert.ok(overviews.length > 1);
  assert.ok(overviews.every((diagram) => diagram.spec.components.length <= 12));
  assert.equal(new Set(overviews.flatMap((diagram) => diagram.memoryIds)).size, 17);
  assert.ok(overviews.every((diagram) => !diagram.spec.components.some(
    (component) => component.label.includes('instance 1'),
  )));
});

test('US1 keeps an all-unconnected System View on non-negative grid rows', () => {
  const unconnected = structuredClone(document);
  unconnected.components = unconnected.components.filter(
    (component) => component.kind === 'memory',
  );
  unconnected.relationships = [];
  const hierarchy = planned(unconnected).find(
    (diagram) => diagram.section === 'system_view',
  );
  assert.ok(hierarchy);
  assert.ok(hierarchy.spec.components.every((component) => component.row >= 0));
  assert.ok(hierarchy.spec.components
    .filter((component) => component.id.startsWith('memory-'))
    .every((component) => component.tag === 'unconnected' || component.tag === 'bank'));
});

test('US1 gallery defaults to System View and exposes only System and Component sections', () => {
  const diagrams = planned();
  const catalog = buildGalleryCatalog(document, diagrams);
  assert.equal(catalog.schema_version, 'mlar.archify-gallery.v1');
  assert.equal(catalog.language, 'en');
  assert.equal(catalog.default_diagram_id, 'system-view-1');
  assert.deepEqual(catalog.sections, [
    { id: 'system_view', label: 'System View' },
    { id: 'component_views', label: 'Component Views' },
  ]);
  const hierarchy = catalog.diagrams.find((diagram) => diagram.id === 'system-view-1');
  assert.deepEqual(hierarchy.memory_names.sort(), ['DRAM', 'L1', 'Scratch', 'Shared', 'Shared'].sort());
  assert.ok(hierarchy.scope_ids.includes('scope-core'));
});

test('US2 creates one focused view for every memory, processor, and data mover', () => {
  const diagrams = planned();
  const focused = diagrams.filter(
    (diagram) => diagram.section === 'component_views' && diagram.focusComponentId,
  );
  assert.deepEqual(
    focused.map((diagram) => diagram.focusComponentId).sort(),
    document.components
      .filter((component) => ['memory', 'processor', 'data_mover'].includes(component.kind))
      .map((component) => component.id)
      .sort(),
  );
  assert.ok(focused.every((diagram) => diagram.componentIds.includes(diagram.focusComponentId)));
});

test('US2 data-mover view combines direct memory routes and required resources', () => {
  const moverView = planned().find(
    (diagram) => diagram.focusComponentId === 'processor-copy',
  );
  assert.ok(moverView);
  assert.equal(moverView.title, 'Data Mover · system / copy');
  assert.deepEqual(moverView.componentIds.sort(), [
    'memory-dram',
    'memory-l1',
    'processor-copy',
    'resource-bus',
  ]);
  assert.deepEqual(moverView.relationshipIds.sort(), [
    'relationship-copy-read',
    'relationship-copy-requires',
    'relationship-copy-write',
  ]);
  assert.ok(!moverView.componentIds.includes('network-noc'), 'transitive neighbors must be absent');
  assert.equal(
    moverView.spec.components.find((component) => component.id === 'processor-copy').type,
    'messagebus',
  );
  assert.equal(
    moverView.spec.connections.find(
      (connection) => connection.id === 'relationship-copy-requires',
    ).label,
    'requires',
  );
  assert.ok(moverView.spec.connections
    .filter((connection) => ['relationship-copy-read', 'relationship-copy-write'].includes(
      connection.id,
    ))
    .every((connection) => !Object.hasOwn(connection, 'label')));
});

test('US2 memory view contains exact direct actors and network attachment only', () => {
  const memoryView = planned().find(
    (diagram) => diagram.focusComponentId === 'memory-l1',
  );
  assert.ok(memoryView);
  assert.equal(memoryView.title, 'Memory · system / core / L1');
  assert.deepEqual(memoryView.componentIds.sort(), [
    'memory-l1',
    'network-noc',
    'processor-copy',
    'processor-lane',
  ]);
  assert.deepEqual(memoryView.relationshipIds.sort(), [
    'relationship-copy-write',
    'relationship-lane-read',
    'relationship-lane-write',
    'relationship-noc-attachment',
  ]);
  assert.ok(!memoryView.componentIds.includes('memory-dram'));
  assert.ok(!memoryView.componentIds.includes('resource-bus'));
});

test('US2 keeps same-memory actor meanings without duplicating identities', () => {
  const sameMemory = structuredClone(document);
  sameMemory.components.push(actor('data_mover', 'processor-loop', 'scope-system', 'loop'));
  sameMemory.relationships.push(
    relationship('relationship-loop-read', 'read', 'memory-dram', 'processor-loop'),
    relationship('relationship-loop-write', 'write', 'processor-loop', 'memory-dram'),
  );
  const view = planned(sameMemory).find(
    (diagram) => diagram.focusComponentId === 'processor-loop',
  );
  assert.ok(view);
  assert.equal(view.componentIds.filter((id) => id === 'processor-loop').length, 1);
  assert.equal(view.componentIds.filter((id) => id === 'memory-dram').length, 1);
  assert.deepEqual(
    view.relationshipIds.filter((id) => id.startsWith('relationship-loop-')).sort(),
    ['relationship-loop-read', 'relationship-loop-write'],
  );
});

test('US2 deterministically partitions more than twelve memory neighbors', () => {
  const large = structuredClone(document);
  for (let index = 0; index < 13; index += 1) {
    const id = `processor-scratch-${index}`;
    large.components.push(actor('processor', id, 'scope-core', `scratch ${index}`));
    large.relationships.push(
      relationship(`relationship-scratch-${index}`, 'read', 'memory-scratch', id),
    );
  }
  const pages = planned(large).filter(
    (diagram) => diagram.focusComponentId === 'memory-scratch',
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
    ['component-memory-scratch-1', 'component-memory-scratch-2'],
  );
});

test('US2 keeps unconnected memories and processors as anchor-only component views', () => {
  const views = planned();
  for (const focusId of ['memory-scratch', 'processor-orphan']) {
    const view = views.find((diagram) => diagram.focusComponentId === focusId);
    assert.ok(view);
    assert.deepEqual(view.componentIds, [focusId]);
    assert.deepEqual(view.relationshipIds, []);
  }
});

test('US2 does not create dedicated resource or network views', () => {
  const focusedIds = planned()
    .filter((diagram) => diagram.focusComponentId)
    .map((diagram) => diagram.focusComponentId);
  assert.ok(!focusedIds.includes('resource-bus'));
  assert.ok(!focusedIds.includes('network-noc'));
});

test('US3 preserves opposite routes in System View', () => {
  const bidirectional = structuredClone(document);
  bidirectional.components.push(
    actor('data_mover', 'processor-copy-back', 'scope-system', 'copy back'),
  );
  bidirectional.relationships.push(
    relationship('relationship-copy-back-read', 'read', 'memory-l1', 'processor-copy-back'),
    relationship('relationship-copy-back-write', 'write', 'processor-copy-back', 'memory-dram'),
  );
  const view = planned(bidirectional).find((diagram) => diagram.section === 'system_view');
  assert.ok(view);
  const componentById = new Map(view.spec.components.map((component) => [component.id, component]));
  for (const actorId of ['processor-copy', 'processor-copy-back']) {
    assert.ok(
      componentById.get('memory-dram').col < componentById.get(actorId).col
        && componentById.get(actorId).col < componentById.get('memory-l1').col,
    );
  }
  assert.deepEqual(
    view.relationshipIds.filter((id) => id.startsWith('relationship-copy-back-')).sort(),
    ['relationship-copy-back-read', 'relationship-copy-back-write'],
  );
});

test('US3 preserves complete canonical coverage through system and component views', () => {
  const diagrams = planned();
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

});

test('US3 groups uncovered components and empty scopes under their architecture scope', () => {
  const uncovered = structuredClone(document);
  uncovered.components.push({
    kind: 'resource',
    id: 'resource-unused',
    scope: 'scope-system',
    name: 'unused resource',
    resource_kind: 'exclusive',
    capacity: null,
  });
  uncovered.scopes.push({
    id: 'scope-empty',
    name: 'empty cluster',
    parent_scope: 'scope-system',
    dimensions: [],
    replication_factor: 1,
  });
  const views = planned(uncovered);
  const componentFallback = views.find(
    (diagram) => diagram.title === 'Architecture Scope · system · unconnected components',
  );
  assert.ok(componentFallback.componentIds.includes('resource-unused'));
  const scopeView = views.find(
    (diagram) => diagram.title === 'Architecture Scope · system / empty cluster',
  );
  assert.ok(scopeView);
  assert.deepEqual(
    scopeView.spec.components.map(({ label, type }) => ({ label, type })),
    [{ label: 'empty cluster', type: 'frontend' }],
  );
});

test('US3 partitions a dense component neighborhood around the stable anchor', () => {
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
    (diagram) => diagram.focusComponentId === 'processor-copy',
  );
  assert.equal(pages.length, 2);
  assert.ok(pages.every((diagram) => diagram.spec.components.length <= 12));
  assert.equal(
    new Set(pages.flatMap((diagram) => diagram.relationshipIds)
      .filter((id) => id.startsWith('relationship-copy-resource-'))).size,
    13,
  );
  assert.ok(pages.every((page) => page.componentIds.includes('processor-copy')));
});

test('US3 planning and component-aware catalog output are deterministic', () => {
  const first = planned();
  const second = planned();
  assert.deepEqual(second, first);
  const catalog = buildGalleryCatalog(document, first);
  assert.deepEqual(
    catalog.sections.map((section) => section.id),
    ['system_view', 'component_views'],
  );
  assert.equal(catalog.default_diagram_id, 'system-view-1');
  const copy = catalog.diagrams.find(
    (diagram) => diagram.focus_component_id === 'processor-copy',
  );
  assert.equal(copy.focus_component_name, 'copy');
  assert.equal(copy.focus_component_kind, 'data_mover');
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
