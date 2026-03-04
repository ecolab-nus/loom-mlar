import { Handle, Position, type NodeProps } from '@xyflow/react';

import type { ArchFlowNodeData } from '../flow';

export function ArchNode({ data }: NodeProps) {
  if (!isArchFlowNodeData(data)) {
    return (
      <div className="arch-node node-memory">
        <div className="arch-node-kind">invalid</div>
        <div className="arch-node-title">Malformed node payload</div>
      </div>
    );
  }

  const nodeData = data;
  const kindClass = nodeData.kind === 'memory' ? 'node-memory' : 'node-processor';

  return (
    <div className={`arch-node ${kindClass}`}>
      <div className="arch-node-kind">{nodeData.kind}</div>
      <div className="arch-node-title">{nodeData.name}</div>
      <div className="arch-node-summary">{nodeData.summary}</div>
      {nodeData.dimensions.length > 0 && (
        <div className="arch-node-dims">
          {nodeData.dimensions.map((dim) => (
            <span key={`${nodeData.name}-${dim.name}`} className="dim-chip">
              {dim.name}:{dim.size_expr}
            </span>
          ))}
        </div>
      )}

      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

function isArchFlowNodeData(value: unknown): value is ArchFlowNodeData {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const candidate = value as Partial<ArchFlowNodeData>;
  return (
    (candidate.kind === 'memory' || candidate.kind === 'processor') &&
    typeof candidate.name === 'string' &&
    typeof candidate.summary === 'string' &&
    Array.isArray(candidate.dimensions)
  );
}
