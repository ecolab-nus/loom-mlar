export type NodeKind = 'memory' | 'processor' | 'router';

export interface GraphDimension {
  name: string;
  size_expr: string;
  size_const: number | null;
}

export interface GraphExpr {
  expr: string;
  const_value: number | null;
}

export interface GraphMlirModuleRef {
  path: string;
  functions: string[];
}

export type GraphProcessorElem =
  | {
      kind: 'unit';
      name: string | null;
      compute: GraphMlirModuleRef | null;
    }
  | {
      kind: 'array';
      name: string | null;
      dimensions: GraphDimension[];
      elem: GraphProcessorElem;
    }
  | {
      kind: 'set';
      name: string | null;
      parts: GraphProcessorElem[];
    };

export type GraphMemoryRegion =
  | {
      kind: 'bank';
      name: string | null;
      capacity_bytes: { expr: string; const_value: number | null };
      access_granularity: { expr: string; const_value: number | null } | null;
      total_size_bytes?: number | null;
    }
  | {
      kind: 'replicated' | 'array';
      name: string | null;
      dimensions: GraphDimension[];
      sub_region: GraphMemoryRegion;
      total_size_bytes?: number | null;
    }
  | {
      kind: 'group';
      name: string | null;
      parts: GraphMemoryRegion[];
      total_size_bytes?: number | null;
    };

export type GraphNodeDetails =
  | {
      type: 'memory';
      region: GraphMemoryRegion;
    }
  | {
      type: 'processor';
      element: GraphProcessorElem;
      total_instances: number | null;
    }
  | {
      type: 'router';
      router: { name: string; side_count: number; endpoints: number };
    };

export interface ArchitectureGraphNode {
  id: string;
  kind: NodeKind;
  name: string;
  label: string;
  dimensions: GraphDimension[];
  details: GraphNodeDetails;
}

export type GraphEdgeDirection = 'directional' | 'bidirectional';

export interface ArchitectureGraphEdge {
  id: string;
  kind: string;
  name: string;
  source: string;
  target: string;
  source_name: string;
  target_name: string;
  label: string;
  direction?: GraphEdgeDirection;
  bandwidth?: GraphExpr;
  latency?: GraphExpr | null;
  constraints?: string;
  sharing?: string;
  map_relation?: 'one_to_one' | 'one_to_many' | 'many_to_one' | 'many_to_many' | 'unknown';
  topology?: 'ring' | 'general';
  map?: {
    source_dimensions: GraphDimension[];
    target_dimensions: GraphDimension[];
    expressions: string[];
  };
  side?: number | null;
}

export interface ArchitectureGraph {
  schema_version: 'mlar.arch-graph.v1';
  architecture: {
    name: string;
    labels: Array<{
      name: string;
      dimensions: GraphDimension[];
    }>;
  };
  nodes: ArchitectureGraphNode[];
  edges: ArchitectureGraphEdge[];
  intra_core?: ArchitectureGraph;
}

// ── Hierarchy schema types ───────────────────────────────────

export type HierarchyNodeKind = 'unit' | 'array' | 'graph' | 'memory' | 'router';

export interface HierarchyRouterEndpoint {
  name: string;
  target_kind: string;
  target_ref: string;
}

export interface HierarchyRouterSide {
  name: string;
  endpoints: HierarchyRouterEndpoint[];
}

export type HierarchyNodeDetails =
  | { type: 'processor'; functions: string[] }
  | { type: 'memory'; region: GraphMemoryRegion; total_size_bytes: number | null }
  | {
      type: 'router';
      router: { name: string; side_count: number; endpoints?: number };
      sides?: HierarchyRouterSide[];
    };

export interface HierarchyConnectivity {
  name: string;
  kind: string;
  bandwidth: GraphExpr;
  latency: GraphExpr | null;
  topology: string;
}

export interface HierarchyNode {
  kind: HierarchyNodeKind;
  name: string;
  dimensions?: GraphDimension[];
  total_instances?: number | null;
  details?: HierarchyNodeDetails | null;
  connectivity?: HierarchyConnectivity[];
  children?: HierarchyNode[];
}

export interface ArchitectureHierarchy {
  schema_version: 'mlar.arch-hierarchy.v1';
  root: HierarchyNode;
}

// ── Viewer (combined) schema types ───────────────────────────

export interface ArchitectureViewer {
  schema_version: 'mlar.arch-viewer.v1';
  hierarchy: HierarchyNode;
  graphs: Record<string, ArchitectureGraph>;
}

// ── Schema detection and parsing ─────────────────────────────

export type AnyArchPayload =
  | { type: 'graph'; data: ArchitectureGraph }
  | { type: 'hierarchy'; data: ArchitectureHierarchy }
  | { type: 'viewer'; data: ArchitectureViewer };

const GRAPH_SCHEMA_VERSION = 'mlar.arch-graph.v1';
const HIERARCHY_SCHEMA_VERSION = 'mlar.arch-hierarchy.v1';
const VIEWER_SCHEMA_VERSION = 'mlar.arch-viewer.v1';

function isRecord(input: unknown): input is Record<string, unknown> {
  return typeof input === 'object' && input !== null;
}

function isStringArray(input: unknown): input is string[] {
  return Array.isArray(input) && input.every((entry) => typeof entry === 'string');
}

function isDimensionArray(input: unknown): input is GraphDimension[] {
  if (!Array.isArray(input)) {
    return false;
  }

  return input.every((dim) => {
    if (!isRecord(dim)) {
      return false;
    }

    return (
      typeof dim.name === 'string' &&
      typeof dim.size_expr === 'string' &&
      (typeof dim.size_const === 'number' || dim.size_const === null)
    );
  });
}

export function parseArchitectureGraph(raw: unknown): ArchitectureGraph {
  if (!isRecord(raw)) {
    throw new Error('Payload must be a JSON object.');
  }

  if (raw.schema_version !== GRAPH_SCHEMA_VERSION) {
    throw new Error(`Expected schema_version=${GRAPH_SCHEMA_VERSION}.`);
  }

  if (!isRecord(raw.architecture) || typeof raw.architecture.name !== 'string') {
    throw new Error('Missing architecture metadata.');
  }

  if (!Array.isArray(raw.nodes) || !Array.isArray(raw.edges)) {
    throw new Error('nodes and edges must be arrays.');
  }

  for (const node of raw.nodes) {
    if (!isRecord(node)) {
      throw new Error('Node entries must be objects.');
    }

    if (typeof node.id !== 'string' || typeof node.name !== 'string') {
      throw new Error('Node must include string id and name.');
    }

    if (node.kind !== 'memory' && node.kind !== 'processor' && node.kind !== 'router') {
      throw new Error(`Invalid node kind for ${node.id}.`);
    }

    if (!isDimensionArray(node.dimensions)) {
      throw new Error(`Node ${node.id} has invalid dimensions.`);
    }
  }

  for (const edge of raw.edges) {
    if (!isRecord(edge)) {
      throw new Error('Edge entries must be objects.');
    }

    if (
      typeof edge.id !== 'string' ||
      typeof edge.source !== 'string' ||
      typeof edge.target !== 'string' ||
      typeof edge.name !== 'string'
    ) {
      throw new Error('Edge must include id/source/target/name.');
    }

    if (
      edge.direction !== undefined &&
      edge.direction !== 'directional' &&
      edge.direction !== 'bidirectional'
    ) {
      throw new Error(`Edge ${edge.id} has invalid direction.`);
    }

    if (!isRecord(edge.map) || !isStringArray(edge.map.expressions)) {
      throw new Error(`Edge ${edge.id} has invalid map.`);
    }
  }

  return raw as unknown as ArchitectureGraph;
}

export function parseArchitectureHierarchy(raw: unknown): ArchitectureHierarchy {
  if (!isRecord(raw)) {
    throw new Error('Payload must be a JSON object.');
  }

  if (raw.schema_version !== HIERARCHY_SCHEMA_VERSION) {
    throw new Error(`Expected schema_version=${HIERARCHY_SCHEMA_VERSION}.`);
  }

  if (!isRecord(raw.root) || typeof raw.root.kind !== 'string' || typeof raw.root.name !== 'string') {
    throw new Error('Missing or invalid root node in hierarchy.');
  }

  return raw as unknown as ArchitectureHierarchy;
}

export function parseArchitectureViewer(raw: unknown): ArchitectureViewer {
  if (!isRecord(raw)) {
    throw new Error('Payload must be a JSON object.');
  }

  if (raw.schema_version !== VIEWER_SCHEMA_VERSION) {
    throw new Error(`Expected schema_version=${VIEWER_SCHEMA_VERSION}.`);
  }

  if (!isRecord(raw.hierarchy) || typeof raw.hierarchy.kind !== 'string' || typeof raw.hierarchy.name !== 'string') {
    throw new Error('Missing or invalid hierarchy node in viewer payload.');
  }

  if (!isRecord(raw.graphs)) {
    throw new Error('Missing graphs map in viewer payload.');
  }

  return raw as unknown as ArchitectureViewer;
}

export function parseAnyArchPayload(raw: unknown): AnyArchPayload {
  if (!isRecord(raw)) {
    throw new Error('Payload must be a JSON object.');
  }

  if (raw.schema_version === VIEWER_SCHEMA_VERSION) {
    return { type: 'viewer', data: parseArchitectureViewer(raw) };
  }

  if (raw.schema_version === HIERARCHY_SCHEMA_VERSION) {
    return { type: 'hierarchy', data: parseArchitectureHierarchy(raw) };
  }

  if (raw.schema_version === GRAPH_SCHEMA_VERSION) {
    return { type: 'graph', data: parseArchitectureGraph(raw) };
  }

  throw new Error(
    `Unknown schema_version: ${String(raw.schema_version)}. ` +
      `Expected ${GRAPH_SCHEMA_VERSION}, ${HIERARCHY_SCHEMA_VERSION}, or ${VIEWER_SCHEMA_VERSION}.`,
  );
}
