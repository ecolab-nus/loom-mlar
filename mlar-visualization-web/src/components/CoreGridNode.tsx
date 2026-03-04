import { Handle, Position, type NodeProps } from '@xyflow/react';

import type { GridFlowNodeData } from '../flow';

export function CoreGridNode({ data }: NodeProps) {
  if (!isGridFlowNodeData(data)) {
    return (
      <div className="arch-node node-memory">
        <div className="arch-node-kind">invalid</div>
        <div className="arch-node-title">Malformed grid payload</div>
      </div>
    );
  }

  const { archName, cols, rows, onCoreClick } = data;

  return (
    <div className="core-grid-node">
      <div className="core-grid-header">
        <span className="core-grid-kind">architecture</span>
        <span className="core-grid-title">{archName}</span>
        <span className="core-grid-subtitle">
          {cols}×{rows} cores — click a core to inspect
        </span>
      </div>

      <div
        className="core-grid"
        style={{
          gridTemplateColumns: `repeat(${cols}, 1fr)`,
          gridTemplateRows: `repeat(${rows}, 1fr)`,
        }}
      >
        {Array.from({ length: rows }, (_, y) =>
          Array.from({ length: cols }, (_, x) => (
            <button
              key={`${x}-${y}`}
              className="core-cell"
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onCoreClick?.(x, y);
              }}
              title={`Core (${x}, ${y})`}
            >
              <span className="core-cell-coord">
                {x},{y}
              </span>
            </button>
          )),
        )}
      </div>

      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

function isGridFlowNodeData(value: unknown): value is GridFlowNodeData {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const c = value as Partial<GridFlowNodeData>;
  return typeof c.archName === 'string' && typeof c.cols === 'number' && typeof c.rows === 'number';
}
