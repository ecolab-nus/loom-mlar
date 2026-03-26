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

function isArchFlowNodeData(value: unknown): value is ArchFlowNodeData {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const candidate = value as Partial<ArchFlowNodeData>;
  return (
    (candidate.kind === 'memory' || candidate.kind === 'processor' || candidate.kind === 'link' || candidate.kind === 'router') &&
    typeof candidate.name === 'string' &&
    typeof candidate.summary === 'string' &&
    Array.isArray(candidate.dimensions) &&
    Array.isArray(candidate.bankSlots)
  );
}

function classForNode(nodeData: ArchFlowNodeData): string {
  if (nodeData.kind === 'memory') {
    return 'node-memory';
  }
  if (nodeData.kind === 'processor') {
    return 'node-processor';
  }
  if (nodeData.kind === 'router') {
    return 'node-router';
  }
  return 'node-link';
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
