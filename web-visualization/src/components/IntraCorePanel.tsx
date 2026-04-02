import {
  Background,
  Controls,
  type Edge,
  type NodeTypes,
  ReactFlow,
  ReactFlowProvider,
} from '@xyflow/react';

import type { ArchitectureGraph, GraphMemoryRegion } from '../schema';
import { architectureToFlow, type ArchFlowNode, type ArchFlowNodeData } from '../flow';
import { ArchNode } from './ArchNode';

interface IntraCorePanelProps {
  coreX: number;
  coreY: number;
  intraCoreGraph: ArchitectureGraph;
  onClose: () => void;
  onMemoryClick?: (name: string, region: GraphMemoryRegion) => void;
}

const innerNodeTypes: NodeTypes = { archNode: ArchNode };

export function IntraCorePanel({
  coreX,
  coreY,
  intraCoreGraph,
  onClose,
  onMemoryClick,
}: IntraCorePanelProps) {
  const { nodes: rawNodes, edges } = architectureToFlow(intraCoreGraph, 'hardware');
  const nodes = rawNodes.filter((node): node is ArchFlowNode => node.type === 'archNode');

  return (
    <div className="intra-core-overlay" onClick={onClose}>
      <div className="intra-core-panel" onClick={(e) => e.stopPropagation()}>
        <header className="intra-core-header">
          <div>
            <h2>
              Core ({coreX}, {coreY})
            </h2>
            <p>{intraCoreGraph.architecture.name} — internal architecture</p>
          </div>
          <button type="button" className="intra-core-close" onClick={onClose}>
            ✕
          </button>
        </header>

        <div className="intra-core-canvas">
          <ReactFlowProvider>
            <ReactFlow<ArchFlowNode, Edge>
              nodes={nodes}
              edges={edges}
              nodeTypes={innerNodeTypes}
              fitView
              fitViewOptions={{ padding: 0.25 }}
              proOptions={{ hideAttribution: true }}
              onNodeClick={(_, node) => {
                const data = node.data as ArchFlowNodeData;
                if (data.kind === 'memory' && data.region && onMemoryClick) {
                  onMemoryClick(data.name, data.region);
                }
              }}
            >
              <Controls />
              <Background color="#cddee8" gap={18} size={1} />
            </ReactFlow>
          </ReactFlowProvider>
        </div>
      </div>
    </div>
  );
}
