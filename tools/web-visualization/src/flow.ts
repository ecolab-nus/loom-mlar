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
  region?: GraphMemoryRegion;
}

export type ArchFlowNode = Node<ArchFlowNodeData, 'archNode'>;

export interface CoreMemorySummary {
  name: string;
  summary: string;
  region?: GraphMemoryRegion;
}

export interface CoreArchNodeData extends Record<string, unknown> {
  coreX: number;
  coreY: number;
  memories: CoreMemorySummary[];
  onMemoryClick?: (name: string, region: GraphMemoryRegion) => void;
}

export type CoreArchFlowNode = Node<CoreArchNodeData, 'coreArchNode'>;

export interface GridFlowNodeData extends Record<string, unknown> {
  archName: string;
  cols: number;
  rows: number;
  onCoreClick?: (x: number, y: number) => void;
}

export type GridFlowNode = Node<GridFlowNodeData, 'coreGridNode'>;

export type AnyFlowNode = ArchFlowNode | CoreArchFlowNode | GridFlowNode;

export interface CoreLinkLegendEntry {
  name: string;
  color: string;
  bandwidth: string;
}

export interface FlowConversionResult {
  nodes: AnyFlowNode[];
  edges: Edge[];
  coreLinkLegend: CoreLinkLegendEntry[];
}

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
  region?: GraphMemoryRegion;
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
const CORE_STEP_X = 248;
const CORE_STEP_Y = 188;

/**
 * Detect whether the graph has 2D grid labels (produced by scale()).
 * Returns { cols, rows } if found, null otherwise.
 */
export function detectGridLayout(
  graph: ArchitectureGraph,
): { cols: number; rows: number; colDim: string; rowDim: string } | null {
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
    colDim: concreteDims[0].name,
    rowDim: concreteDims[1].name,
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
  onMemoryClick?: (name: string, region: GraphMemoryRegion) => void,
): FlowConversionResult {
  const grid = detectGridLayout(graph);

  if (grid && graph.intra_core) {
    const coreLevel = buildCoreLevelFlow(graph, grid, onMemoryClick);
    if (coreLevel) {
      return coreLevel;
    }

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

    return { nodes: [gridNode], edges: [], coreLinkLegend: [] };
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
        region: node.region,
      },
      draggable: true,
    };
  });

  return {
    nodes,
    edges: visual.edges.map((edge) => edgeToFlow(edge)),
    coreLinkLegend: [],
  };
}

function buildCoreLevelFlow(
  graph: ArchitectureGraph,
  grid: { cols: number; rows: number; colDim: string; rowDim: string },
  onMemoryClick?: (name: string, region: GraphMemoryRegion) => void,
): FlowConversionResult | null {
  const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
  const intraMemoryByName = new Map<string, ArchitectureGraphNode>();
  if (graph.intra_core) {
    for (const node of graph.intra_core.nodes) {
      if (node.kind === 'memory' && node.details.type === 'memory') {
        intraMemoryByName.set(node.name, node);
      }
    }
  }

  const interCoreEdges = graph.edges.filter((edge) => {
    const sourceNode = nodeById.get(edge.source);
    const targetNode = nodeById.get(edge.target);
    if (!sourceNode || !targetNode) {
      return false;
    }
    if (sourceNode.kind !== 'memory' || targetNode.kind !== 'memory') {
      return false;
    }
    if (!isOneToOneMap(edge)) {
      return false;
    }
    return (
      edge.map.source_dimensions.some((d) => d.name === grid.colDim) &&
      edge.map.source_dimensions.some((d) => d.name === grid.rowDim) &&
      edge.map.target_dimensions.some((d) => d.name === grid.colDim) &&
      edge.map.target_dimensions.some((d) => d.name === grid.rowDim)
    );
  });

  if (interCoreEdges.length === 0) {
    return null;
  }

  const memoryByName = new Map<string, CoreMemorySummary>();
  for (const edge of interCoreEdges) {
    const source = nodeById.get(edge.source);
    const target = nodeById.get(edge.target);
    if (source && source.kind === 'memory' && !memoryByName.has(source.name)) {
      memoryByName.set(source.name, summarizeCoreMemory(intraMemoryByName.get(source.name) ?? source));
    }
    if (target && target.kind === 'memory' && !memoryByName.has(target.name)) {
      memoryByName.set(target.name, summarizeCoreMemory(intraMemoryByName.get(target.name) ?? target));
    }
  }
  const memories = Array.from(memoryByName.values()).sort((a, b) => a.name.localeCompare(b.name));

  const nodes: CoreArchFlowNode[] = [];
  for (let y = 0; y < grid.rows; y += 1) {
    for (let x = 0; x < grid.cols; x += 1) {
      nodes.push({
        id: coreNodeId(x, y),
        type: 'coreArchNode',
        position: {
          x: 80 + x * CORE_STEP_X,
          y: 80 + y * CORE_STEP_Y,
        },
        data: {
          coreX: x,
          coreY: y,
          memories,
          onMemoryClick,
        },
        draggable: true,
      });
    }
  }

  const nodeIds = new Set(nodes.map((node) => node.id));
  const edgeKeys = new Set<string>();
  const edges: Edge[] = [];
  const legendByName = new Map<string, CoreLinkLegendEntry>();
  for (const edge of interCoreEdges) {
    const color = colorForInterCoreEdge(edge.name);
    if (!legendByName.has(edge.name)) {
      legendByName.set(edge.name, {
        name: edge.name,
        color,
        bandwidth: edge.bandwidth.expr,
      });
    }
    const sourceNode = nodeById.get(edge.source);
    const targetNode = nodeById.get(edge.target);
    if (!sourceNode || !targetNode || sourceNode.kind !== 'memory' || targetNode.kind !== 'memory') {
      continue;
    }
    const sourceMemory = sourceNode.name;
    const targetMemory = targetNode.name;
    const pairs = enumerateCoreMappings(edge, grid);
    for (const pair of pairs) {
      const source = coreNodeId(pair.sourceX, pair.sourceY);
      const target = coreNodeId(pair.targetX, pair.targetY);
      if (!nodeIds.has(source) || !nodeIds.has(target)) {
        continue;
      }
      const key = `${source}->${target}:${edge.name}`;
      if (edgeKeys.has(key)) {
        continue;
      }
      edgeKeys.add(key);
      const handles = directionalHandles(pair.sourceX, pair.sourceY, pair.targetX, pair.targetY);
      edges.push({
        id: `core-edge:${edge.name}:${pair.sourceX},${pair.sourceY}:${pair.targetX},${pair.targetY}`,
        type: 'straight',
        source,
        target,
        sourceHandle: memoryHandleId(sourceMemory, 'source', handles.sourceSide),
        targetHandle: memoryHandleId(targetMemory, 'target', handles.targetSide),
        zIndex: 1000,
        markerEnd: {
          type: MarkerType.ArrowClosed,
        },
        style: {
          stroke: color,
          strokeWidth: 1.2,
        },
      });
    }
  }

  if (edges.length === 0) {
    return null;
  }

  const coreLinkLegend = Array.from(legendByName.values()).sort((a, b) => a.name.localeCompare(b.name));

  return { nodes, edges, coreLinkLegend };
}

function isOneToOneMap(edge: ArchitectureGraphEdge): boolean {
  if (edge.map_relation === 'one_to_one') {
    return true;
  }
  const srcCard = productConcreteSizes(edge.map.source_dimensions);
  const dstCard = productConcreteSizes(edge.map.target_dimensions);
  return srcCard !== null && dstCard !== null && srcCard === dstCard;
}

function coreNodeId(x: number, y: number): string {
  return `core|${x}|${y}`;
}

function enumerateCoreMappings(
  edge: ArchitectureGraphEdge,
  grid: { cols: number; rows: number; colDim: string; rowDim: string },
): Array<{ sourceX: number; sourceY: number; targetX: number; targetY: number }> {
  const sourceDims = edge.map.source_dimensions;
  const assignments = enumerateAssignments(sourceDims);
  if (!assignments) {
    return [];
  }

  const out: Array<{ sourceX: number; sourceY: number; targetX: number; targetY: number }> = [];
  for (const sourceAssignment of assignments) {
    const sourceX = sourceAssignment[grid.colDim];
    const sourceY = sourceAssignment[grid.rowDim];
    if (
      sourceX === undefined ||
      sourceY === undefined ||
      sourceX < 0 ||
      sourceY < 0 ||
      sourceX >= grid.cols ||
      sourceY >= grid.rows
    ) {
      continue;
    }

    const targetAssignment: Record<string, number> = {};
    let failed = false;
    for (let i = 0; i < edge.map.expressions.length; i += 1) {
      const targetDim = edge.map.target_dimensions[i];
      const value = evaluateAffineExpression(edge.map.expressions[i], sourceAssignment);
      if (value === null) {
        failed = true;
        break;
      }
      targetAssignment[targetDim.name] = value;
    }
    if (failed) {
      continue;
    }

    const targetX = targetAssignment[grid.colDim];
    const targetY = targetAssignment[grid.rowDim];
    if (
      targetX === undefined ||
      targetY === undefined ||
      targetX < 0 ||
      targetY < 0 ||
      targetX >= grid.cols ||
      targetY >= grid.rows
    ) {
      continue;
    }

    out.push({ sourceX, sourceY, targetX, targetY });
  }

  return out;
}

function enumerateAssignments(dimensions: GraphDimension[]): Array<Record<string, number>> | null {
  const concrete = dimensions.map((dim) => ({ name: dim.name, size: dim.size_const }));
  if (concrete.some((dim) => dim.size === null)) {
    return null;
  }

  const out: Array<Record<string, number>> = [];
  const current = new Array(concrete.length).fill(0);

  const walk = (depth: number) => {
    if (depth === concrete.length) {
      const assignment: Record<string, number> = {};
      for (let i = 0; i < concrete.length; i += 1) {
        assignment[concrete[i].name] = current[i];
      }
      out.push(assignment);
      return;
    }

    for (let value = 0; value < concrete[depth].size!; value += 1) {
      current[depth] = value;
      walk(depth + 1);
    }
  };

  walk(0);
  return out;
}

function evaluateAffineExpression(expr: string, vars: Record<string, number>): number | null {
  let trimmed = expr.trim();
  while (isWrappedByOuterParentheses(trimmed)) {
    trimmed = trimmed.slice(1, -1).trim();
  }

  if (/^-?\d+$/.test(trimmed)) {
    return Number.parseInt(trimmed, 10);
  }
  if (Object.prototype.hasOwnProperty.call(vars, trimmed)) {
    return vars[trimmed];
  }

  const operators = [' mod ', ' ceildiv ', ' + ', ' * '] as const;
  for (const op of operators) {
    const parts = splitTopLevel(trimmed, op);
    if (!parts) {
      continue;
    }
    const left = evaluateAffineExpression(parts[0], vars);
    const right = evaluateAffineExpression(parts[1], vars);
    if (left === null || right === null) {
      return null;
    }

    if (op === ' mod ') {
      if (right === 0) {
        return null;
      }
      return ((left % right) + right) % right;
    }
    if (op === ' ceildiv ') {
      if (right === 0) {
        return null;
      }
      return Math.floor((left + right - 1) / right);
    }
    if (op === ' + ') {
      return left + right;
    }
    if (op === ' * ') {
      return left * right;
    }
  }

  return null;
}

function directionalHandles(
  sourceX: number,
  sourceY: number,
  targetX: number,
  targetY: number,
): { sourceSide: 'north' | 'south' | 'east' | 'west'; targetSide: 'north' | 'south' | 'east' | 'west' } {
  if (sourceX === targetX) {
    if (targetY > sourceY) {
      return { sourceSide: 'south', targetSide: 'north' };
    }
    return { sourceSide: 'north', targetSide: 'south' };
  }

  if (sourceY === targetY) {
    if (targetX > sourceX) {
      return { sourceSide: 'east', targetSide: 'west' };
    }
    return { sourceSide: 'west', targetSide: 'east' };
  }

  const dx = Math.abs(targetX - sourceX);
  const dy = Math.abs(targetY - sourceY);
  if (dx >= dy) {
    return targetX >= sourceX
      ? { sourceSide: 'east', targetSide: 'west' }
      : { sourceSide: 'west', targetSide: 'east' };
  }
  return targetY >= sourceY
    ? { sourceSide: 'south', targetSide: 'north' }
    : { sourceSide: 'north', targetSide: 'south' };
}

function memoryHandleId(
  memoryName: string,
  kind: 'source' | 'target',
  side: 'north' | 'south' | 'east' | 'west',
): string {
  return `${slugify(memoryName)}-${kind}-${side}`;
}

function splitTopLevel(expr: string, token: string): [string, string] | null {
  let depth = 0;
  for (let i = 0; i <= expr.length - token.length; i += 1) {
    const ch = expr[i];
    if (ch === '(') {
      depth += 1;
      continue;
    }
    if (ch === ')') {
      depth -= 1;
      continue;
    }
    if (depth === 0 && expr.slice(i, i + token.length) === token) {
      const left = expr.slice(0, i).trim();
      const right = expr.slice(i + token.length).trim();
      return [left, right];
    }
  }
  return null;
}

function isWrappedByOuterParentheses(expr: string): boolean {
  if (!(expr.startsWith('(') && expr.endsWith(')'))) {
    return false;
  }
  let depth = 0;
  for (let i = 0; i < expr.length; i += 1) {
    const ch = expr[i];
    if (ch === '(') {
      depth += 1;
    } else if (ch === ')') {
      depth -= 1;
      if (depth === 0 && i < expr.length - 1) {
        return false;
      }
    }
  }
  return depth === 0;
}

function colorForInterCoreEdge(name: string): string {
  if (name.includes('_x')) {
    return '#cf7a3d';
  }
  if (name.includes('_y')) {
    return '#2d6aa2';
  }
  return '#3e6d89';
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
        region: node.details.region,
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

export function countBankLeaves(region: GraphMemoryRegion): number | null {
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
    case 'replicated':
    case 'array': {
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

function summarizeCoreMemory(node: ArchitectureGraphNode): CoreMemorySummary {
  const described = describeNode(node).visualNode;
  return {
    name: described.name,
    summary: described.region ? compactRegionSummary(described.region) : described.summary,
    region: described.region,
  };
}

function compactRegionSummary(region: GraphMemoryRegion): string {
  switch (region.kind) {
    case 'bank': {
      const size = regionTotalBytes(region);
      return size !== null ? formatBytesCompact(size) : region.capacity_bytes.expr;
    }
    case 'replicated':
    case 'array': {
      const n = productConcreteSizes(region.dimensions);
      const elemSize = regionTotalBytes(region.elem);
      if (n !== null && elemSize !== null) {
        return `${n} × ${formatBytesCompact(elemSize)}`;
      }
      const dimStr = region.dimensions.map((d) => `${d.name}=${d.size_expr}`).join(', ');
      return `replicated [${dimStr}]`;
    }
    case 'group': {
      const total = regionTotalBytes(region);
      if (total !== null) {
        return `${region.parts.length} parts, ${formatBytesCompact(total)}`;
      }
      return `${region.parts.length} parts`;
    }
  }
}

function regionTotalBytes(region: GraphMemoryRegion): number | null {
  if (typeof region.total_size_bytes === 'number') {
    return region.total_size_bytes;
  }
  switch (region.kind) {
    case 'bank':
      return region.capacity_bytes.const_value;
    case 'replicated':
    case 'array': {
      const elemSize = regionTotalBytes(region.elem);
      const multiplier = productConcreteSizes(region.dimensions);
      return elemSize !== null && multiplier !== null ? elemSize * multiplier : null;
    }
    case 'group': {
      let total = 0;
      for (const part of region.parts) {
        const partSize = regionTotalBytes(part);
        if (partSize === null) return null;
        total += partSize;
      }
      return total;
    }
  }
}

function formatBytesCompact(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) {
    const val = bytes / (1024 * 1024 * 1024);
    return `${Number.isInteger(val) ? val : val.toFixed(1)} GB`;
  }
  if (bytes >= 1024 * 1024) {
    const val = bytes / (1024 * 1024);
    return `${Number.isInteger(val) ? val : val.toFixed(1)} MB`;
  }
  if (bytes >= 1024) {
    const val = bytes / 1024;
    return `${Number.isInteger(val) ? val : val.toFixed(1)} KB`;
  }
  return `${bytes} B`;
}

export function productConcreteSizes(dimensions: GraphDimension[]): number | null {
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
  if (node.kind === 'router' && node.details.type === 'router') {
    return `${node.details.router.side_count} sides`;
  }
  return '';
}

function summarizeMemoryNode(node: ArchitectureGraphNode, totalBanks: number | null): string {
  if (node.kind !== 'memory' || node.details.type !== 'memory') {
    return summarizeArchitectureNode(node);
  }
  if (totalBanks === null) {
    return `banks=symbolic`;
  }
  return `banks=${totalBanks}`;
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
