import { Handle, Position, type NodeProps } from '@xyflow/react';

import type { ArchFlowNodeData, BankSlot } from '../flow';

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
  const kindClass = classForNode(nodeData);
  const showBankPreview = nodeData.kind === 'memory' && nodeData.bankSlots.length > 0;

  if (nodeData.kind === 'resource') {
    return (
      <div className="arch-node-resource-wrapper">
        <div className="node-resource-hexagon">
          <div className="arch-node-kind">resource</div>
          <div className="arch-node-title">{nodeData.name}</div>
          <div className="arch-node-summary">{nodeData.summary}</div>
        </div>
        <Handle type="target" position={Position.Left} />
        <Handle type="source" position={Position.Right} />
      </div>
    );
  }

  const kindLabel = nodeData.kind === 'data_mover' ? 'data mover' : nodeData.kind;

  return (
    <div className={`arch-node ${kindClass}`}>
      <div className="arch-node-kind">{kindLabel}</div>
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

      {showBankPreview && (
        <div className="memory-bank-preview">
          {nodeData.bankSlots.map((slot) => (
            <BankSlotChip key={`${nodeData.name}-${slot.id}`} slot={slot} />
          ))}
        </div>
      )}

      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

const VALID_KINDS = new Set([
  'memory', 'processor', 'data_mover', 'array', 'graph', 'link', 'router', 'resource',
]);

function isArchFlowNodeData(value: unknown): value is ArchFlowNodeData {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const candidate = value as Partial<ArchFlowNodeData>;
  return (
    typeof candidate.kind === 'string' &&
    VALID_KINDS.has(candidate.kind) &&
    typeof candidate.name === 'string' &&
    typeof candidate.summary === 'string' &&
    Array.isArray(candidate.dimensions) &&
    Array.isArray(candidate.bankSlots)
  );
}

function classForNode(nodeData: ArchFlowNodeData): string {
  switch (nodeData.kind) {
    case 'memory':
      return 'node-memory';
    case 'processor':
      return 'node-processor';
    case 'data_mover':
      return 'node-data-mover';
    case 'array':
      return 'node-array';
    case 'graph':
      return 'node-graph';
    case 'router':
      return 'node-router';
    case 'resource':
      return 'node-resource';
    default:
      return 'node-link';
  }
}

function BankSlotChip({ slot }: { slot: BankSlot }) {
  const slotClass = slot.isOverflow ? 'bank-slot bank-slot-overflow' : 'bank-slot';
  const title =
    slot.hiddenCount !== null
      ? `Hidden banks: ${slot.hiddenCount}`
      : slot.isOverflow
        ? 'More banks'
        : `Bank ${slot.label}`;

  return (
    <div className={slotClass} title={title}>
      <span>{slot.label}</span>
    </div>
  );
}
