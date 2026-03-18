import { useCallback, useState } from 'react';
import type {
  ArchitectureHierarchy,
  HierarchyNode,
  HierarchyNodeKind,
  HierarchyConnectivity,
  HierarchyRouterSide,
  GraphDimension,
  GraphMemoryRegion,
} from '../schema';

interface HierarchyViewProps {
  hierarchy: ArchitectureHierarchy;
}

export function HierarchyView({ hierarchy }: HierarchyViewProps) {
  return (
    <div className="hierarchy-view">
      <div className="hierarchy-header">
        <span className="hierarchy-badge">hierarchy</span>
        <h2 className="hierarchy-title">{hierarchy.root.name}</h2>
        <span className="hierarchy-subtitle">Architecture Hierarchy</span>
      </div>
      <div className="hierarchy-tree">
        <HierarchyTreeNode node={hierarchy.root} depth={0} isLast />
      </div>
    </div>
  );
}

interface HierarchyTreeNodeProps {
  node: HierarchyNode;
  depth: number;
  isLast: boolean;
}

function HierarchyTreeNode({ node, depth, isLast }: HierarchyTreeNodeProps) {
  const hasChildren = (node.children?.length ?? 0) > 0;
  const [expanded, setExpanded] = useState(depth < 3);

  const toggle = useCallback(() => {
    if (hasChildren) {
      setExpanded((v) => !v);
    }
  }, [hasChildren]);

  const children = node.children ?? [];

  return (
    <div className={`htree-node ${isLast ? 'htree-node--last' : ''}`}>
      <div className="htree-node-row" onClick={toggle}>
        {hasChildren && (
          <span className={`htree-toggle ${expanded ? 'htree-toggle--open' : ''}`}>
            {expanded ? '\u25BE' : '\u25B8'}
          </span>
        )}
        {!hasChildren && <span className="htree-toggle htree-toggle--leaf" />}
        <span className={`htree-kind htree-kind--${node.kind}`}>{kindLabel(node.kind)}</span>
        <span className="htree-name">{node.name}</span>
        {(node.dimensions?.length ?? 0) > 0 && (
          <span className="htree-dims">
            {node.dimensions!.map((d) => `${d.name}=${d.size_expr}`).join(', ')}
          </span>
        )}
        {node.total_instances != null && node.total_instances > 1 && (
          <span className="htree-instances">{node.total_instances} instances</span>
        )}
        <NodeSummaryBadge node={node} />
      </div>

      {expanded && (
        <div className="htree-children">
          {node.connectivity && node.connectivity.length > 0 && (
            <ConnectivityBlock connectivity={node.connectivity} />
          )}
          {node.details && <DetailsBlock details={node.details} />}
          {children.map((child, idx) => (
            <HierarchyTreeNode
              key={`${child.kind}-${child.name}-${idx}`}
              node={child}
              depth={depth + 1}
              isLast={idx === children.length - 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function NodeSummaryBadge({ node }: { node: HierarchyNode }) {
  if (node.kind === 'memory' && node.details?.type === 'memory') {
    const total = node.details.total_size_bytes;
    if (total != null) {
      return <span className="htree-size">{formatBytesCompact(total)}</span>;
    }
  }
  if (node.kind === 'unit' && node.details?.type === 'processor') {
    const count = node.details.functions.length;
    if (count > 0) {
      return (
        <span className="htree-func-count">
          {count} fn{count > 1 ? 's' : ''}
        </span>
      );
    }
  }
  if (node.kind === 'router' && node.details?.type === 'router') {
    return (
      <span className="htree-router-info">
        {node.details.router.side_count} sides, {node.details.router.endpoints} endpoints
      </span>
    );
  }
  return null;
}

function ConnectivityBlock({ connectivity }: { connectivity: HierarchyConnectivity[] }) {
  return (
    <div className="htree-connectivity">
      <div className="htree-section-label">Connectivity</div>
      {connectivity.map((conn) => (
        <div key={conn.name} className="htree-conn-item">
          <span className={`htree-conn-badge htree-conn-badge--${conn.topology}`}>
            {conn.topology}
          </span>
          <span className="htree-conn-name">{conn.name}</span>
          <span className="htree-conn-bw">bw={conn.bandwidth.expr}</span>
          {conn.latency && (
            <span className="htree-conn-lat">lat={conn.latency.expr}</span>
          )}
        </div>
      ))}
    </div>
  );
}

interface DetailsBlockProps {
  details: NonNullable<HierarchyNode['details']>;
}

function DetailsBlock({ details }: DetailsBlockProps) {
  if (details.type === 'processor') {
    if (details.functions.length === 0) return null;
    return (
      <div className="htree-details">
        <div className="htree-section-label">Functions</div>
        <div className="htree-func-list">
          {details.functions.map((fn) => (
            <span key={fn} className="htree-func-chip">
              {fn}
            </span>
          ))}
        </div>
      </div>
    );
  }

  if (details.type === 'memory') {
    return (
      <div className="htree-details">
        <MemoryRegionSummary region={details.region} />
      </div>
    );
  }

  if (details.type === 'router') {
    return (
      <div className="htree-details">
        <div className="htree-section-label">Router Sides</div>
        {details.sides.map((side) => (
          <RouterSideBlock key={side.name} side={side} />
        ))}
      </div>
    );
  }

  return null;
}

function MemoryRegionSummary({ region }: { region: GraphMemoryRegion }) {
  if (region.kind === 'bank') {
    return (
      <div className="htree-mem-bank">
        <span className="htree-mem-label">bank</span>
        <span className="htree-mem-value">capacity={region.capacity_bytes.expr}</span>
        {region.access_granularity && (
          <span className="htree-mem-value">block={region.access_granularity.expr}</span>
        )}
      </div>
    );
  }

  if (region.kind === 'replicated' || region.kind === 'array') {
    const dimStr = (region.dimensions ?? []).map((d: GraphDimension) => `${d.name}=${d.size_expr}`).join(', ');
    return (
      <div className="htree-mem-array">
        <span className="htree-mem-label">array [{dimStr}]</span>
        {region.total_size_bytes != null && (
          <span className="htree-mem-value">total={formatBytesCompact(region.total_size_bytes)}</span>
        )}
      </div>
    );
  }

  return null;
}

function RouterSideBlock({ side }: { side: HierarchyRouterSide }) {
  const [expanded, setExpanded] = useState(false);
  const count = side.endpoints.length;

  return (
    <div className="htree-router-side">
      <div className="htree-router-side-header" onClick={() => setExpanded((v) => !v)}>
        <span className={`htree-toggle htree-toggle--small ${expanded ? 'htree-toggle--open' : ''}`}>
          {expanded ? '\u25BE' : '\u25B8'}
        </span>
        <span className="htree-router-side-name">{side.name}</span>
        <span className="htree-router-side-count">{count} endpoint{count !== 1 ? 's' : ''}</span>
      </div>
      {expanded && (
        <div className="htree-router-endpoints">
          {side.endpoints.map((ep) => (
            <div key={ep.name} className="htree-router-ep">
              <span className={`htree-ep-kind htree-ep-kind--${ep.target_kind}`}>
                {ep.target_kind}
              </span>
              <span className="htree-ep-name">{ep.name}</span>
              <span className="htree-ep-ref">{ep.target_ref}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function kindLabel(kind: HierarchyNodeKind): string {
  switch (kind) {
    case 'unit':
      return 'processor';
    case 'array':
      return 'array';
    case 'graph':
      return 'graph';
    case 'memory':
      return 'memory';
    case 'router':
      return 'router';
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
