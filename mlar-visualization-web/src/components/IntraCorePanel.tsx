import {
  Background,
  Controls,
  type Edge,
  type NodeTypes,
  ReactFlow,
  ReactFlowProvider,
} from '@xyflow/react';

import type { ArchitectureGraph } from '../schema';
import { architectureToFlow, type ArchFlowNode } from '../flow';
import { ArchNode } from './ArchNode';

interface IntraCorePanelProps {
  coreX: number;
  coreY: number;
  intraCoreGraph: ArchitectureGraph;
  onClose: () => void;
}

const innerNodeTypes: NodeTypes = { archNode: ArchNode };

export function IntraCorePanel({ coreX, coreY, intraCoreGraph, onClose }: IntraCorePanelProps) {
  const { nodes: rawNodes, edges } = architectureToFlow(intraCoreGraph);
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
