import { Handle, Position, type NodeProps } from '@xyflow/react';

import type { BankSlot, CoreArchNodeData, CoreMemorySummary } from '../flow';

export function CoreArchNode({ data }: NodeProps) {
  if (!isCoreArchNodeData(data)) {
    return (
      <div className="core-arch-node">
        <div className="core-arch-kind">invalid</div>
        <div className="core-arch-title">Malformed core payload</div>
      </div>
    );
  }

  const { coreX, coreY, memories } = data;

  return (
    <div className="core-arch-node">
      <div className="core-arch-kind">core</div>
      <div className="core-arch-title">
        ({coreX}, {coreY})
      </div>
      <div className="core-arch-subtitle">architecture</div>

      <div className="core-arch-mems">
        {memories.map((memory) => (
          <div key={`${coreX}-${coreY}-${memory.name}`} className="core-memory-block">
            <span className="core-memory-name">{memory.name}</span>
            <div className="core-memory-summary">{memory.summary}</div>

            {memory.dimensions.length > 0 && (
              <div className="core-memory-dims">
                {memory.dimensions.map((dim) => (
                  <span key={`${memory.name}-${dim.name}`} className="dim-chip">
                    {dim.name}:{dim.size_expr}
                  </span>
                ))}
              </div>
            )}

            {memory.bankSlots.length > 0 && (
              <div className="memory-bank-preview">
                {memory.bankSlots.map((slot) => (
                  <BankSlotChip key={`${memory.name}-${slot.id}`} slot={slot} />
                ))}
              </div>
            )}

            <Handle
              type="source"
              id={`${slugify(memory.name)}-source-north`}
              position={Position.Top}
              className="core-direction-handle"
              style={{ left: '42%' }}
            />
            <Handle
              type="target"
              id={`${slugify(memory.name)}-target-north`}
              position={Position.Top}
              className="core-direction-handle"
              style={{ left: '58%' }}
            />
            <Handle
              type="source"
              id={`${slugify(memory.name)}-source-south`}
              position={Position.Bottom}
              className="core-direction-handle"
              style={{ left: '42%' }}
            />
            <Handle
              type="target"
              id={`${slugify(memory.name)}-target-south`}
              position={Position.Bottom}
              className="core-direction-handle"
              style={{ left: '58%' }}
            />
            <Handle
              type="source"
              id={`${slugify(memory.name)}-source-east`}
              position={Position.Right}
              className="core-direction-handle"
              style={{ top: '42%' }}
            />
            <Handle
              type="target"
              id={`${slugify(memory.name)}-target-east`}
              position={Position.Right}
              className="core-direction-handle"
              style={{ top: '58%' }}
            />
            <Handle
              type="source"
              id={`${slugify(memory.name)}-source-west`}
              position={Position.Left}
              className="core-direction-handle"
              style={{ top: '42%' }}
            />
            <Handle
              type="target"
              id={`${slugify(memory.name)}-target-west`}
              position={Position.Left}
              className="core-direction-handle"
              style={{ top: '58%' }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

function isCoreArchNodeData(value: unknown): value is CoreArchNodeData {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const c = value as Partial<CoreArchNodeData>;
  return (
    typeof c.coreX === 'number' &&
    typeof c.coreY === 'number' &&
    Array.isArray(c.memories) &&
    c.memories.every(isCoreMemorySummary)
  );
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

function isCoreMemorySummary(value: unknown): value is CoreMemorySummary {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const mem = value as Partial<CoreMemorySummary>;
  return (
    typeof mem.name === 'string' &&
    typeof mem.summary === 'string' &&
    Array.isArray(mem.dimensions) &&
    Array.isArray(mem.bankSlots)
  );
}

function slugify(value: string): string {
  const out = value
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return out.length > 0 ? out : 'mem';
}
