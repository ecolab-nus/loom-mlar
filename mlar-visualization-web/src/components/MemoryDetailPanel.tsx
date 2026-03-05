import { Fragment, useState } from 'react';

import type { GraphMemoryRegion, GraphDimension } from '../schema';
import { countBankLeaves, productConcreteSizes } from '../flow';

interface MemoryDetailPanelProps {
  name: string;
  region: GraphMemoryRegion;
  onClose: () => void;
}

interface BreadcrumbEntry {
  name: string;
  region: GraphMemoryRegion;
}

export function MemoryDetailPanel({ name, region, onClose }: MemoryDetailPanelProps) {
  const [path, setPath] = useState<BreadcrumbEntry[]>([]);

  const currentRegion = path.length > 0 ? path[path.length - 1].region : region;
  const currentName = path.length > 0 ? path[path.length - 1].name : name;

  const drillDown = (childName: string, childRegion: GraphMemoryRegion) => {
    setPath([...path, { name: childName, region: childRegion }]);
  };

  const navigateTo = (index: number) => {
    setPath(path.slice(0, index));
  };

  return (
    <div className="memory-detail-overlay" onClick={onClose}>
      <div className="memory-detail-panel" onClick={(e) => e.stopPropagation()}>
        <header className="memory-detail-header">
          <div>
            <h2>{currentName}</h2>
            <p>Memory region detail</p>
            {path.length > 0 && (
              <nav className="memory-breadcrumb">
                <button type="button" onClick={() => navigateTo(0)}>
                  {name}
                </button>
                {path.map((entry, i) => (
                  <Fragment key={i}>
                    <span className="memory-breadcrumb-sep">/</span>
                    <button type="button" onClick={() => navigateTo(i + 1)}>
                      {entry.name}
                    </button>
                  </Fragment>
                ))}
              </nav>
            )}
          </div>
          <button type="button" className="memory-detail-close" onClick={onClose}>
            ✕
          </button>
        </header>

        <div className="memory-detail-content">
          <RegionView region={currentRegion} onDrillDown={drillDown} />
        </div>
      </div>
    </div>
  );
}

function RegionView({
  region,
  onDrillDown,
}: {
  region: GraphMemoryRegion;
  onDrillDown: (name: string, region: GraphMemoryRegion) => void;
}) {
  switch (region.kind) {
    case 'bank':
      return <BankView region={region} />;
    case 'replicated':
      return <ReplicatedView region={region} onDrillDown={onDrillDown} />;
    case 'group':
      return <GroupView region={region} onDrillDown={onDrillDown} />;
  }
}

function BankView({
  region,
}: {
  region: Extract<GraphMemoryRegion, { kind: 'bank' }>;
}) {
  const capacity = region.capacity_bytes;
  const granularity = region.access_granularity;
  const blocks =
    capacity.const_value !== null && granularity?.const_value
      ? capacity.const_value / granularity.const_value
      : null;

  return (
    <div className="memory-region-section">
      <div className="region-type-badge region-type-bank">bank</div>
      {region.name && <div className="region-section-name">{region.name}</div>}
      <div className="bank-properties">
        <div className="bank-prop">
          <span className="bank-prop-label">Capacity</span>
          <code className="bank-prop-value">{formatBytesExpr(capacity)}</code>
        </div>
        {granularity && (
          <div className="bank-prop">
            <span className="bank-prop-label">Block size</span>
            <code className="bank-prop-value">{formatBytesExpr(granularity)}</code>
          </div>
        )}
        {blocks !== null && (
          <div className="bank-prop">
            <span className="bank-prop-label">Blocks per bank</span>
            <code className="bank-prop-value">{blocks.toLocaleString()}</code>
          </div>
        )}
      </div>
    </div>
  );
}

function ReplicatedView({
  region,
  onDrillDown,
}: {
  region: Extract<GraphMemoryRegion, { kind: 'replicated' }>;
  onDrillDown: (name: string, region: GraphMemoryRegion) => void;
}) {
  const copies = productConcreteSizes(region.dimensions);
  const totalBanks = countBankLeaves(region);
  const isLeafChild = region.elem.kind === 'bank';

  return (
    <div className="memory-region-section">
      <div className="region-type-badge region-type-replicated">replicated</div>
      {region.name && <div className="region-section-name">{region.name}</div>}

      <div className="region-info-grid">
        <DimensionChips dimensions={region.dimensions} />
        {copies !== null && (
          <div className="region-stat">
            <span className="region-stat-label">Copies</span>
            <span className="region-stat-value">{copies}</span>
          </div>
        )}
        {totalBanks !== null && (
          <div className="region-stat">
            <span className="region-stat-label">Total banks</span>
            <span className="region-stat-value">{totalBanks}</span>
          </div>
        )}
      </div>

      <div className="region-children-section">
        <h3 className="region-children-heading">
          {isLeafChild ? 'Bank properties' : 'Element'}
          {copies !== null && isLeafChild && (
            <span className="region-children-note"> (each of {copies} banks)</span>
          )}
        </h3>
        {isLeafChild ? (
          <BankView region={region.elem as Extract<GraphMemoryRegion, { kind: 'bank' }>} />
        ) : (
          <SubRegionCard region={region.elem} onClick={onDrillDown} />
        )}
      </div>
    </div>
  );
}

function GroupView({
  region,
  onDrillDown,
}: {
  region: Extract<GraphMemoryRegion, { kind: 'group' }>;
  onDrillDown: (name: string, region: GraphMemoryRegion) => void;
}) {
  const totalBanks = countBankLeaves(region);

  return (
    <div className="memory-region-section">
      <div className="region-type-badge region-type-group">group</div>
      {region.name && <div className="region-section-name">{region.name}</div>}

      <div className="region-info-grid">
        <div className="region-stat">
          <span className="region-stat-label">Parts</span>
          <span className="region-stat-value">{region.parts.length}</span>
        </div>
        {totalBanks !== null && (
          <div className="region-stat">
            <span className="region-stat-label">Total banks</span>
            <span className="region-stat-value">{totalBanks}</span>
          </div>
        )}
      </div>

      <div className="region-children-section">
        <h3 className="region-children-heading">Sub-regions</h3>
        <div className="region-children-list">
          {region.parts.map((part, i) =>
            part.kind === 'bank' ? (
              <BankView key={i} region={part} />
            ) : (
              <SubRegionCard key={i} region={part} onClick={onDrillDown} />
            ),
          )}
        </div>
      </div>
    </div>
  );
}

function SubRegionCard({
  region,
  onClick,
}: {
  region: GraphMemoryRegion;
  onClick: (name: string, region: GraphMemoryRegion) => void;
}) {
  const label = regionSummaryLabel(region);
  const displayName = region.name ?? regionKindLabel(region);
  const banks = countBankLeaves(region);

  return (
    <button
      type="button"
      className="sub-region-card"
      onClick={() => onClick(displayName, region)}
    >
      <div className="sub-region-card-top">
        <span className={`region-type-badge region-type-${region.kind}`}>{region.kind}</span>
        <span className="sub-region-card-name">{displayName}</span>
      </div>
      <div className="sub-region-card-info">{label}</div>
      {banks !== null && banks > 1 && (
        <div className="sub-region-card-banks">{banks} banks</div>
      )}
      <span className="sub-region-card-arrow">→</span>
    </button>
  );
}

function DimensionChips({ dimensions }: { dimensions: GraphDimension[] }) {
  if (dimensions.length === 0) return null;
  return (
    <div className="region-dim-chips">
      {dimensions.map((dim) => (
        <span key={dim.name} className="dim-chip">
          {dim.name}:{dim.size_expr}
        </span>
      ))}
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${bytes} B`;
}

function formatBytesExpr(expr: { expr: string; const_value: number | null }): string {
  if (expr.const_value !== null) {
    const nice = formatBytes(expr.const_value);
    if (nice !== `${expr.const_value} B`) {
      return `${nice} (${expr.const_value.toLocaleString()} B)`;
    }
    return `${expr.const_value} B`;
  }
  return expr.expr;
}

function regionKindLabel(region: GraphMemoryRegion): string {
  switch (region.kind) {
    case 'bank':
      return 'bank';
    case 'replicated':
      return 'replicated';
    case 'group':
      return 'group';
  }
}

function regionSummaryLabel(region: GraphMemoryRegion): string {
  switch (region.kind) {
    case 'bank': {
      const cap = region.capacity_bytes.const_value;
      return cap !== null ? `bank, ${formatBytes(cap)}` : `bank, capacity=${region.capacity_bytes.expr}`;
    }
    case 'replicated': {
      const copies = productConcreteSizes(region.dimensions);
      const dimStr = region.dimensions.map((d) => `${d.name}=${d.size_expr}`).join(', ');
      return copies !== null ? `${copies}× replicated [${dimStr}]` : `replicated [${dimStr}]`;
    }
    case 'group':
      return `group of ${region.parts.length} parts`;
  }
}
