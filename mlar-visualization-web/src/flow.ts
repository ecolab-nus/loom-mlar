import type { Edge, Node } from '@xyflow/react';
import { MarkerType } from '@xyflow/react';

import type {
  ArchitectureGraph,
  ArchitectureGraphEdge,
  ArchitectureGraphNode,
  GraphDimension,
  NodeKind,
} from './schema';

export interface ArchFlowNodeData extends Record<string, unknown> {
  kind: NodeKind;
  name: string;
  label: string;
  dimensions: GraphDimension[];
  summary: string;
}

export type ArchFlowNode = Node<ArchFlowNodeData, 'archNode'>;

export interface GridFlowNodeData extends Record<string, unknown> {
  archName: string;
  cols: number;
  rows: number;
  onCoreClick?: (x: number, y: number) => void;
}

export type GridFlowNode = Node<GridFlowNodeData, 'coreGridNode'>;

export type AnyFlowNode = ArchFlowNode | GridFlowNode;

interface LayoutResult {
  levels: Map<string, number>;
  lanes: Map<number, ArchitectureGraphNode[]>;
}

const LANE_WIDTH = 340;
const ROW_HEIGHT = 170;

/**
 * Detect whether the graph has 2D grid labels (produced by scale()).
 * Returns { cols, rows } if found, null otherwise.
 */
export function detectGridLayout(
  graph: ArchitectureGraph,
): { cols: number; rows: number } | null {
  if (!graph.architecture.labels || graph.architecture.labels.length === 0) {
    return null;
  }

  // Collect all label dimensions.
  const allDims = graph.architecture.labels.flatMap((l) => l.dimensions);
  if (allDims.length < 2) {
    return null;
  }

  // Need at least two concrete dimensions.
  const concreteDims = allDims.filter((d) => d.size_const !== null);
  if (concreteDims.length < 2) {
    return null;
  }

  // Use first two dimensions as cols × rows.
  return {
    cols: concreteDims[0].size_const!,
    rows: concreteDims[1].size_const!,
  };
}

/**
 * Convert an architecture graph to React Flow nodes/edges.
 *
 * When the graph has 2D labels (from scale()), produces a single grid node.
 * Otherwise falls back to the flat per-node layout.
 */
export function architectureToFlow(
  graph: ArchitectureGraph,
  onCoreClick?: (x: number, y: number) => void,
): {
  nodes: AnyFlowNode[];
  edges: Edge[];
} {
  const grid = detectGridLayout(graph);

  if (grid && graph.intra_core) {
    // Grid mode: single large node showing the 2D array of cores.
    const gridNode: GridFlowNode = {
      id: 'core-grid',
      type: 'coreGridNode',
      position: { x: 80, y: 80 },
      data: {
        archName: graph.architecture.name,
        cols: grid.cols,
        rows: grid.rows,
        onCoreClick,
      },
      draggable: true,
    };

    return { nodes: [gridNode], edges: [] };
  }

  // Flat mode: existing layout.
  const { levels, lanes } = buildLayout(graph);

  const nodes: ArchFlowNode[] = graph.nodes.map((node) => {
    const level = levels.get(node.id) ?? 0;
    const laneNodes = lanes.get(level) ?? [];
    const row = laneNodes.findIndex((n) => n.id === node.id);

    return {
      id: node.id,
      type: 'archNode',
      position: {
        x: 80 + level * LANE_WIDTH,
        y: 80 + Math.max(0, row) * ROW_HEIGHT,
      },
      data: {
        kind: node.kind,
        name: node.name,
        label: node.label,
        dimensions: node.dimensions,
        summary: summarizeNode(node),
      },
      draggable: true,
    };
  });

  const edges: Edge[] = graph.edges.map((edge) => edgeToFlow(edge));

  return { nodes, edges };
}

function buildLayout(graph: ArchitectureGraph): LayoutResult {
  const outgoing = new Map<string, string[]>();
  const indegree = new Map<string, number>();

  for (const node of graph.nodes) {
    outgoing.set(node.id, []);
    indegree.set(node.id, 0);
  }

  for (const edge of graph.edges) {
    const list = outgoing.get(edge.source);
    if (list) {
      list.push(edge.target);
    }

    const current = indegree.get(edge.target);
    if (current !== undefined) {
      indegree.set(edge.target, current + 1);
    }
  }

  const queue: string[] = [];
  const level = new Map<string, number>();

  for (const [nodeId, inCount] of indegree.entries()) {
    if (inCount === 0) {
      queue.push(nodeId);
      level.set(nodeId, 0);
    }
  }

  let queueIndex = 0;
  while (queueIndex < queue.length) {
    const nodeId = queue[queueIndex++];
    const srcLevel = level.get(nodeId) ?? 0;

    for (const target of outgoing.get(nodeId) ?? []) {
      level.set(target, Math.max(level.get(target) ?? 0, srcLevel + 1));
      indegree.set(target, Math.max(0, (indegree.get(target) ?? 0) - 1));
      if ((indegree.get(target) ?? 0) === 0) {
        queue.push(target);
      }
    }
  }

  for (const node of graph.nodes) {
    if (!level.has(node.id)) {
      level.set(node.id, 0);
    }
  }

  const lanes = new Map<number, ArchitectureGraphNode[]>();
  for (const node of graph.nodes) {
    const lane = level.get(node.id) ?? 0;
    if (!lanes.has(lane)) {
      lanes.set(lane, []);
    }
    lanes.get(lane)?.push(node);
  }

  for (const laneNodes of lanes.values()) {
    laneNodes.sort((a, b) => {
      if (a.kind === b.kind) {
        return a.name.localeCompare(b.name);
      }
      return a.kind === 'memory' ? -1 : 1;
    });
  }

  return { levels: level, lanes };
}

function summarizeNode(node: ArchitectureGraphNode): string {
  if (node.kind === 'memory' && node.details.type === 'memory') {
    const qty = node.details.resource.quantity;
    return `resource=${qty}`;
  }

  if (node.kind === 'processor' && node.details.type === 'processor') {
    return node.details.total_instances !== null
      ? `instances=${node.details.total_instances}`
      : 'instances=symbolic';
  }

  return '';
}

function edgeToFlow(edge: ArchitectureGraphEdge): Edge {
  return {
    id: edge.id,
    source: edge.source,
    target: edge.target,
    label: edge.name,
    markerEnd: {
      type: MarkerType.ArrowClosed,
    },
    style: {
      stroke: '#345f78',
      strokeWidth: 1.8,
    },
    labelStyle: {
      fontSize: 11,
      fontWeight: 600,
      fill: '#18313f',
    },
  };
}
