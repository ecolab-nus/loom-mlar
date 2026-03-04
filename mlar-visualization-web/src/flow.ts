import type { Edge, Node } from '@xyflow/react';
import { MarkerType } from '@xyflow/react';

import type {
  ArchitectureGraph,
  ArchitectureGraphEdge,
  ArchitectureGraphNode,
  GraphDimension,
  GraphMemoryRegion,
  NodeKind,
} from './schema';

export type VisualNodeKind = NodeKind | 'link';

export interface BankSlot {
  id: string;
  label: string;
  isOverflow: boolean;
  hiddenCount: number | null;
}

export interface ArchFlowNodeData extends Record<string, unknown> {
  kind: VisualNodeKind;
  name: string;
  label: string;
  dimensions: GraphDimension[];
  summary: string;
  bankSlots: BankSlot[];
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
  lanes: Map<number, VisualNodeSpec[]>;
}

interface VisualNodeSpec {
  id: string;
  kind: VisualNodeKind;
  name: string;
  label: string;
  dimensions: GraphDimension[];
  summary: string;
  bankSlots: BankSlot[];
}

interface EndpointStat {
  nodeId: string;
  multiplicity: number | null;
  sortKey: number;
}

interface VisualEdgeSpec {
  id: string;
  source: string;
  target: string;
  label?: string;
  kind: 'link_in' | 'link_out';
}

interface VisualGraph {
  nodes: VisualNodeSpec[];
  edges: VisualEdgeSpec[];
}

interface LinkAccumulator {
  edge: ArchitectureGraphEdge;
  sources: Map<string, EndpointStat>;
  targets: Map<string, EndpointStat>;
}

const LANE_WIDTH = 320;
const ROW_HEIGHT = 180;
const PREVIEW_HEAD_BANKS = 4;

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

  const allDims = graph.architecture.labels.flatMap((l) => l.dimensions);
  if (allDims.length < 2) {
    return null;
  }

  const concreteDims = allDims.filter((d) => d.size_const !== null);
  if (concreteDims.length < 2) {
    return null;
  }

  return {
    cols: concreteDims[0].size_const!,
    rows: concreteDims[1].size_const!,
  };
}

/**
 * Convert an architecture graph to React Flow nodes/edges.
 *
 * When the graph has 2D labels (from scale()), produces a single grid node.
 * Otherwise renders explicit link blocks and memory-bank previews.
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

  const visual = buildVisualGraph(graph);
  const { levels, lanes } = buildLayout(visual.nodes, visual.edges);

  const nodes: ArchFlowNode[] = visual.nodes.map((node) => {
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
        summary: node.summary,
        bankSlots: node.bankSlots,
      },
      draggable: true,
    };
  });

  return {
    nodes,
    edges: visual.edges.map((edge) => edgeToFlow(edge)),
  };
}

function buildVisualGraph(graph: ArchitectureGraph): VisualGraph {
  const nodes: VisualNodeSpec[] = [];
  const edges: VisualEdgeSpec[] = [];
  const edgeKeys = new Set<string>();
  const endpointStatsByNode = new Map<string, EndpointStat>();

  for (const node of graph.nodes) {
    const described = describeNode(node);
    nodes.push(described.visualNode);
    endpointStatsByNode.set(node.id, {
      nodeId: node.id,
      multiplicity: described.multiplicity,
      sortKey: 0,
    });
  }

  const links = new Map<string, LinkAccumulator>();
  for (const edge of graph.edges) {
    if (!links.has(edge.name)) {
      links.set(edge.name, {
        edge,
        sources: new Map<string, EndpointStat>(),
        targets: new Map<string, EndpointStat>(),
      });
    }
    const acc = links.get(edge.name)!;

    const source = endpointStatsByNode.get(edge.source) ?? defaultEndpointStat(edge.source);
    acc.sources.set(source.nodeId, source);

    const target = endpointStatsByNode.get(edge.target) ?? defaultEndpointStat(edge.target);
    acc.targets.set(target.nodeId, target);
  }

  for (const [name, acc] of links.entries()) {
    const linkNodeId = `link:${slugify(name)}`;
    const latency = acc.edge.latency ? `, lat=${acc.edge.latency.expr}` : '';
    nodes.push({
      id: linkNodeId,
      kind: 'link',
      name,
      label: name,
      dimensions: [],
      summary: `bw=${acc.edge.bandwidth.expr}${latency}`,
      bankSlots: [],
    });

    const sources = Array.from(acc.sources.values()).sort(endpointSort);
    const targets = Array.from(acc.targets.values()).sort(endpointSort);

    for (const source of sources) {
      const multiplicityLabel = formatMultiplicityLabel(source.multiplicity);
      pushEdge(
        edges,
        edgeKeys,
        {
          id: `edge:${source.nodeId}:to:${linkNodeId}`,
          source: source.nodeId,
          target: linkNodeId,
          label: multiplicityLabel,
          kind: 'link_in',
        },
      );
    }

    for (const target of targets) {
      pushEdge(
        edges,
        edgeKeys,
        {
          id: `edge:${linkNodeId}:to:${target.nodeId}`,
          source: linkNodeId,
          target: target.nodeId,
          kind: 'link_out',
        },
      );
    }
  }

  return { nodes, edges };
}

function describeNode(node: ArchitectureGraphNode): {
  visualNode: VisualNodeSpec;
  multiplicity: number | null;
} {
  if (node.kind === 'memory' && node.details.type === 'memory') {
    const totalBanks = countBankLeaves(node.details.region);
    const bankSlots = buildBankSlots(totalBanks);
    const summary = summarizeMemoryNode(node, totalBanks);

    return {
      visualNode: {
        id: node.id,
        kind: node.kind,
        name: node.name,
        label: node.label,
        dimensions: node.dimensions,
        summary,
        bankSlots,
      },
      multiplicity: totalBanks,
    };
  }

  return {
    visualNode: {
      id: node.id,
      kind: node.kind,
      name: node.name,
      label: node.label,
      dimensions: node.dimensions,
      summary: summarizeArchitectureNode(node),
      bankSlots: [],
    },
    multiplicity: 1,
  };
}

function buildBankSlots(totalBanks: number | null): BankSlot[] {
  if (totalBanks === 1) {
    return [];
  }

  if (totalBanks === null) {
    const slots: BankSlot[] = [];
    for (let i = 0; i < PREVIEW_HEAD_BANKS; i += 1) {
      slots.push({
        id: `b${i}`,
        label: `b${i}`,
        isOverflow: false,
        hiddenCount: null,
      });
    }
    slots.push({
      id: 'ellipsis',
      label: '...',
      isOverflow: true,
      hiddenCount: null,
    });
    return slots;
  }

  const visible = Math.min(totalBanks, PREVIEW_HEAD_BANKS);
  const slots: BankSlot[] = [];
  for (let i = 0; i < visible; i += 1) {
    slots.push({
      id: `b${i}`,
      label: `b${i}`,
      isOverflow: false,
      hiddenCount: null,
    });
  }

  if (totalBanks > PREVIEW_HEAD_BANKS + 1) {
    slots.push({
      id: 'ellipsis',
      label: '...',
      isOverflow: true,
      hiddenCount: totalBanks - (PREVIEW_HEAD_BANKS + 1),
    });
    slots.push({
      id: `b${totalBanks - 1}`,
      label: `b${totalBanks - 1}`,
      isOverflow: false,
      hiddenCount: null,
    });
  } else if (totalBanks === PREVIEW_HEAD_BANKS + 1) {
    slots.push({
      id: `b${totalBanks - 1}`,
      label: `b${totalBanks - 1}`,
      isOverflow: false,
      hiddenCount: null,
    });
  }

  return slots;
}

function countBankLeaves(region: GraphMemoryRegion): number | null {
  switch (region.kind) {
    case 'bank':
      return 1;
    case 'group': {
      let sum = 0;
      for (const part of region.parts) {
        const count = countBankLeaves(part);
        if (count === null) {
          return null;
        }
        sum += count;
      }
      return sum;
    }
    case 'replicated': {
      const inner = countBankLeaves(region.elem);
      if (inner === null) {
        return null;
      }
      const multiplier = productConcreteSizes(region.dimensions);
      if (multiplier === null) {
        return null;
      }
      return inner * multiplier;
    }
    default:
      return null;
  }
}

function productConcreteSizes(dimensions: GraphDimension[]): number | null {
  let out = 1;
  for (const dim of dimensions) {
    if (dim.size_const === null) {
      return null;
    }
    out *= dim.size_const;
  }
  return out;
}

function defaultEndpointStat(nodeId: string): EndpointStat {
  return { nodeId, multiplicity: 1, sortKey: 0 };
}

function formatMultiplicityLabel(multiplicity: number | null): string | undefined {
  if (multiplicity === null) {
    return 'x ?';
  }
  if (multiplicity <= 1) {
    return undefined;
  }
  return `x ${multiplicity}`;
}

function endpointSort(a: EndpointStat, b: EndpointStat): number {
  if (a.sortKey !== b.sortKey) {
    return a.sortKey - b.sortKey;
  }
  return a.nodeId.localeCompare(b.nodeId);
}

function pushEdge(edges: VisualEdgeSpec[], keys: Set<string>, edge: VisualEdgeSpec): void {
  const key = `${edge.source}->${edge.target}`;
  if (keys.has(key)) {
    return;
  }
  keys.add(key);
  edges.push(edge);
}

function buildLayout(nodes: VisualNodeSpec[], edges: VisualEdgeSpec[]): LayoutResult {
  const outgoing = new Map<string, string[]>();
  const indegree = new Map<string, number>();

  for (const node of nodes) {
    outgoing.set(node.id, []);
    indegree.set(node.id, 0);
  }

  for (const edge of edges) {
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

  for (const node of nodes) {
    if (!level.has(node.id)) {
      level.set(node.id, 0);
    }
  }

  const lanes = new Map<number, VisualNodeSpec[]>();
  for (const node of nodes) {
    const lane = level.get(node.id) ?? 0;
    if (!lanes.has(lane)) {
      lanes.set(lane, []);
    }
    lanes.get(lane)?.push(node);
  }

  for (const laneNodes of lanes.values()) {
    laneNodes.sort((a, b) => {
      const rankDelta = kindRank(a) - kindRank(b);
      if (rankDelta !== 0) {
        return rankDelta;
      }
      return a.name.localeCompare(b.name);
    });
  }

  return { levels: level, lanes };
}

function kindRank(node: VisualNodeSpec): number {
  if (node.kind === 'memory') {
    return 0;
  }
  if (node.kind === 'link') {
    return 1;
  }
  return 2;
}

function summarizeArchitectureNode(node: ArchitectureGraphNode): string {
  if (node.kind === 'processor' && node.details.type === 'processor') {
    return node.details.total_instances !== null
      ? `instances=${node.details.total_instances}`
      : 'instances=symbolic';
  }
  if (node.kind === 'memory' && node.details.type === 'memory') {
    return `resource=${node.details.resource.quantity}`;
  }
  return '';
}

function summarizeMemoryNode(node: ArchitectureGraphNode, totalBanks: number | null): string {
  if (node.kind !== 'memory' || node.details.type !== 'memory') {
    return summarizeArchitectureNode(node);
  }
  const resource = node.details.resource.quantity;
  if (totalBanks === null) {
    return `resource=${resource}, banks=symbolic`;
  }
  return `resource=${resource}, banks=${totalBanks}`;
}

function edgeToFlow(edge: VisualEdgeSpec): Edge {
  const stroke = edge.kind === 'link_in' ? '#3e6d89' : '#1f5d80';
  return {
    id: edge.id,
    source: edge.source,
    target: edge.target,
    label: edge.label,
    markerEnd: {
      type: MarkerType.ArrowClosed,
    },
    style: {
      stroke,
      strokeWidth: 1.8,
    },
    labelStyle: {
      fontSize: 11,
      fontWeight: 650,
      fill: '#1f4b62',
    },
  };
}

function slugify(value: string): string {
  const out = value
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return out.length > 0 ? out : 'unnamed';
}
