import assert from 'node:assert/strict';
import test from 'node:test';

import { planDiagrams, validateReferences } from '../bin/mlar-archify.mjs';
import { buildGalleryCatalog, renderGalleryHtml } from '../lib/gallery.mjs';

const document = {
  schema_version: 'mlar.visualization.v1',
  architecture: { id: 'architecture-demo', name: 'demo', root_scope: 'scope-demo' },
  scopes: [
    { id: 'scope-demo', name: 'demo', dimensions: [], replication_factor: 1 },
  ],
  components: [
    {
      kind: 'memory',
      id: 'memory-l1',
      scope: 'scope-demo',
      name: 'L1',
      dimensions: [],
      region: {
        kind: 'bank',
        name: 'L1',
        capacity: { text: '1024', constant: 1024 },
        block_size: null,
        total_size_bytes: 1024,
      },
      total_size_bytes: 1024,
    },
    {
      kind: 'processor',
      id: 'processor-lane',
      scope: 'scope-demo',
      name: 'lane',
      effect: 'transform',
      functions: ['add'],
    },
  ],
  relationships: [
    {
      id: 'relationship-read',
      kind: 'read',
      source: 'memory-l1',
      target: 'processor-lane',
      label: 'read',
    },
  ],
};

test('planner creates independent showcase-sized diagrams without collapsed nodes', () => {
  assert.doesNotThrow(() => validateReferences(document));
  const diagrams = planDiagrams(document);
  assert.ok(diagrams.length >= 2);
  assert.ok(diagrams.every((diagram) => diagram.spec.components.length <= 12));
  assert.ok(diagrams.every((diagram) => diagram.spec.meta.quality_profile === 'showcase'));
  assert.ok(diagrams.some((diagram) => diagram.section === 'overview'));
  assert.ok(diagrams.some((diagram) => diagram.section === 'memory_reads'));
  assert.ok(
    diagrams.some((diagram) => diagram.relationshipIds.includes('relationship-read')),
  );
  const allIds = diagrams.flatMap((diagram) => diagram.spec.components.map((component) => component.id));
  assert.ok(allIds.includes('scope-demo'));
  assert.ok(allIds.includes('memory-l1'));
  assert.ok(allIds.includes('processor-lane'));
});

test('gallery provides a root-first web app without drawing its own diagrams', () => {
  const diagrams = planDiagrams(document);
  const catalog = buildGalleryCatalog(document, diagrams);
  assert.equal(catalog.schema_version, 'mlar.archify-gallery.v1');
  assert.equal(catalog.language, 'en');
  assert.equal(catalog.default_diagram_id, 'scope-demo');
  assert.ok(catalog.diagrams.every((diagram) => diagram.html.startsWith('html/')));

  const html = renderGalleryHtml(document, diagrams);
  assert.match(html, /id="diagramFrame"/);
  assert.match(html, /System and subsystems/);
  assert.match(html, /<html lang="en">/);
  assert.doesNotMatch(html, /\p{Script=Han}/u);
  assert.match(html, /Archify 2\.14/);
  assert.doesNotMatch(html, /<svg\b/);
  const script = html.match(/<script>([\s\S]*)<\/script>/)?.[1];
  assert.ok(script);
  assert.doesNotThrow(() => new Function(script));
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
