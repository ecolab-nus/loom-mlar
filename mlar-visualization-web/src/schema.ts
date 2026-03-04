export type NodeKind = 'memory' | 'processor';

export interface GraphDimension {
  name: string;
  size_expr: string;
  size_const: number | null;
}

export interface GraphExpr {
  expr: string;
  const_value: number | null;
}

export interface GraphResource {
  name: string;
  quantity: number;
}

export interface GraphResourceReq {
  resource: GraphResource;
  quantity: number;
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
      resources: GraphResourceReq[];
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
    }
  | {
      kind: 'replicated';
      name: string | null;
      dimensions: GraphDimension[];
      elem: GraphMemoryRegion;
    }
  | {
      kind: 'group';
      name: string | null;
      parts: GraphMemoryRegion[];
    };

export type GraphNodeDetails =
  | {
      type: 'memory';
      region: GraphMemoryRegion;
      resource: GraphResource;
    }
  | {
      type: 'processor';
      element: GraphProcessorElem;
      total_instances: number | null;
    };

export interface ArchitectureGraphNode {
  id: string;
  kind: NodeKind;
  name: string;
  label: string;
  dimensions: GraphDimension[];
  details: GraphNodeDetails;
}

export interface ArchitectureGraphEdge {
  id: string;
  kind: 'link';
  name: string;
  source: string;
  target: string;
  source_name: string;
  target_name: string;
  label: string;
  bandwidth: GraphExpr;
  latency: GraphExpr | null;
  constraints: string;
  sharing: string;
  map_relation?: 'one_to_one' | 'one_to_many' | 'many_to_one' | 'many_to_many' | 'unknown';
  topology?: 'ring' | 'general';
  map: {
    source_dimensions: GraphDimension[];
    target_dimensions: GraphDimension[];
    expressions: string[];
  };
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

const SCHEMA_VERSION = 'mlar.arch-graph.v1';

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

  if (raw.schema_version !== SCHEMA_VERSION) {
    throw new Error(`Expected schema_version=${SCHEMA_VERSION}.`);
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

    if (node.kind !== 'memory' && node.kind !== 'processor') {
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

    if (!isRecord(edge.map) || !isStringArray(edge.map.expressions)) {
      throw new Error(`Edge ${edge.id} has invalid map.`);
    }
  }

  return raw as unknown as ArchitectureGraph;
}
