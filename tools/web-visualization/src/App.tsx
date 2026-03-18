import { useCallback, useEffect, useMemo, useState, type ChangeEvent } from 'react';
import {
  Background,
  Controls,
  type Edge,
  MiniMap,
  type NodeTypes,
  Panel,
  ReactFlow,
  ReactFlowProvider,
} from '@xyflow/react';

import '@xyflow/react/dist/style.css';

import { ArchNode } from './components/ArchNode';
import { CoreArchNode } from './components/CoreArchNode';
import { CoreGridNode } from './components/CoreGridNode';
import { HierarchyView } from './components/HierarchyView';
import { IntraCorePanel } from './components/IntraCorePanel';
import { MemoryDetailPanel } from './components/MemoryDetailPanel';
import {
  architectureToFlow,
  type AnyFlowNode,
  type ArchFlowNodeData,
  type FlowConversionResult,
} from './flow';
import { loadGraphFromFile, loadGraphFromUrl, parseAnyText } from './runtime-loader';
import type {
  ArchitectureGraph,
  ArchitectureViewer,
  HierarchyNode,
  AnyArchPayload,
  GraphMemoryRegion,
} from './schema';

const nodeTypes: NodeTypes = {
  archNode: ArchNode,
  coreArchNode: CoreArchNode,
  coreGridNode: CoreGridNode,
};
const DEFAULT_URL = '/sample-viewer.json';

function parseCoreNodeId(nodeId: string): { x: number; y: number } | null {
  const parts = nodeId.split('|');
  if (parts[0] !== 'core') {
    return null;
  }
  if (parts.length < 3) {
    return null;
  }
  const x = Number.parseInt(parts[parts.length - 2], 10);
  const y = Number.parseInt(parts[parts.length - 1], 10);
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    return null;
  }
  return { x, y };
}

interface ParsedState {
  payload: AnyArchPayload | null;
  hierarchy: HierarchyNode | null;
  graphs: Record<string, ArchitectureGraph>;
  error: string | null;
}

function extractViewData(payload: AnyArchPayload): {
  hierarchy: HierarchyNode | null;
  graphs: Record<string, ArchitectureGraph>;
} {
  switch (payload.type) {
    case 'viewer':
      return {
        hierarchy: payload.data.hierarchy,
        graphs: payload.data.graphs,
      };
    case 'hierarchy':
      return {
        hierarchy: payload.data.root,
        graphs: {},
      };
    case 'graph':
      return {
        hierarchy: null,
        graphs: { '': payload.data },
      };
  }
}

function AppInner() {
  const [jsonText, setJsonText] = useState('');
  const [sourceUrl, setSourceUrl] = useState(DEFAULT_URL);
  const [sourceName, setSourceName] = useState('none');
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isEditorVisible, setIsEditorVisible] = useState(false);
  const [selectedGraphPath, setSelectedGraphPath] = useState<string>('');
  const [selectedCore, setSelectedCore] = useState<{ x: number; y: number } | null>(null);
  const [selectedLegendLinkName, setSelectedLegendLinkName] = useState<string | null>(null);
  const [selectedMemory, setSelectedMemory] = useState<{
    name: string;
    region: GraphMemoryRegion;
  } | null>(null);

  const loadFromUrl = async (url: string) => {
    try {
      const loaded = await loadGraphFromUrl(url);
      setJsonText(loaded.text);
      setSourceName(loaded.source);
      setLoadError(null);
      setSelectedGraphPath('');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load URL';
      setLoadError(message);
    }
  };

  useEffect(() => {
    void loadFromUrl(DEFAULT_URL);
  }, []);

  const parsed = useMemo((): ParsedState => {
    if (!jsonText.trim()) {
      return { payload: null, hierarchy: null, graphs: {}, error: loadError ?? 'No JSON loaded yet.' };
    }

    try {
      const loaded = parseAnyText(jsonText, sourceName);
      const { hierarchy, graphs } = extractViewData(loaded.payload);
      return { payload: loaded.payload, hierarchy, graphs, error: null };
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Invalid JSON payload';
      return { payload: null, hierarchy: null, graphs: {}, error: message };
    }
  }, [jsonText, sourceName, loadError]);

  const availableGraphPaths = useMemo(
    () => new Set(Object.keys(parsed.graphs)),
    [parsed.graphs],
  );

  const activeGraph = useMemo((): ArchitectureGraph | null => {
    return parsed.graphs[selectedGraphPath] ?? null;
  }, [parsed.graphs, selectedGraphPath]);

  const onNodeSelect = useCallback((path: string) => {
    setSelectedGraphPath(path);
    setSelectedCore(null);
    setSelectedMemory(null);
  }, []);

  const onCoreClick = useCallback((x: number, y: number) => {
    setSelectedCore({ x, y });
  }, []);

  const onMemoryClick = useCallback((name: string, region: GraphMemoryRegion) => {
    setSelectedMemory({ name, region });
  }, []);

  const flow = useMemo(() => {
    if (!activeGraph) {
      return {
        nodes: [] as AnyFlowNode[],
        edges: [] as Edge[],
        coreLinkLegend: [],
      } satisfies FlowConversionResult;
    }
    return architectureToFlow(activeGraph, onCoreClick, onMemoryClick);
  }, [activeGraph, onCoreClick, onMemoryClick]);

  useEffect(() => {
    if (flow.coreLinkLegend.length === 0) {
      setSelectedLegendLinkName(null);
      return;
    }
    const hasSelected = flow.coreLinkLegend.some((entry) => entry.name === selectedLegendLinkName);
    if (!hasSelected) {
      setSelectedLegendLinkName(flow.coreLinkLegend[0].name);
    }
  }, [flow.coreLinkLegend, selectedLegendLinkName]);

  const selectedLegendLink = useMemo(
    () => flow.coreLinkLegend.find((entry) => entry.name === selectedLegendLinkName) ?? null,
    [flow.coreLinkLegend, selectedLegendLinkName],
  );

  const onUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    try {
      const loaded = await loadGraphFromFile(file);
      setJsonText(loaded.text);
      setSourceName(loaded.source);
      setLoadError(null);
      setSelectedGraphPath('');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load file';
      setLoadError(message);
    }
  };

  const graphPathLabel = selectedGraphPath === '' ? 'root' : selectedGraphPath;
  const archName = activeGraph?.architecture.name ?? parsed.hierarchy?.name ?? 'no selection';

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>MLAR Architecture Viewer</h1>
          <p>Runtime JSON loader for architecture visualization</p>
        </div>

        <div className="source-controls">
          <input
            className="source-input"
            value={sourceUrl}
            onChange={(event) => setSourceUrl(event.target.value)}
            spellCheck={false}
            placeholder="/sample-viewer.json"
          />
          <button type="button" className="action-button" onClick={() => void loadFromUrl(sourceUrl)}>
            Load URL
          </button>
          <label className="upload-button">
            Open File
            <input type="file" accept="application/json" onChange={onUpload} />
          </label>
          <button
            type="button"
            className="action-button"
            onClick={() => setIsEditorVisible((visible) => !visible)}
          >
            {isEditorVisible ? 'Hide JSON' : 'Show JSON'}
          </button>
        </div>
      </header>

      <main className="app-main-split">
        {parsed.hierarchy && (
          <aside className="hierarchy-sidebar">
            <HierarchyView
              hierarchy={parsed.hierarchy}
              selectedPath={selectedGraphPath}
              availableGraphPaths={availableGraphPaths}
              onNodeSelect={onNodeSelect}
            />
          </aside>
        )}

        <section className="graph-panel">
          {activeGraph ? (
            <ReactFlow<AnyFlowNode, Edge>
              key={selectedGraphPath}
              nodes={flow.nodes}
              edges={flow.edges}
              nodeTypes={nodeTypes}
              fitView
              fitViewOptions={{ padding: 0.15 }}
              onNodeClick={(_, node) => {
                const coord = parseCoreNodeId(node.id);
                if (coord) {
                  setSelectedCore(coord);
                  return;
                }
                if (node.type === 'archNode') {
                  const data = node.data as ArchFlowNodeData;
                  if (data.kind === 'memory' && data.region) {
                    setSelectedMemory({ name: data.name, region: data.region });
                  }
                }
              }}
            >
              <Controls />
              <MiniMap pannable zoomable />
              <Background color="#dbe8ef" gap={22} size={1} />
              <Panel position="top-left">
                <div className="meta-panel">
                  <div>
                    <strong>Architecture</strong>
                    <span>{archName}</span>
                  </div>
                  <div>
                    <strong>Path</strong>
                    <span>{graphPathLabel}</span>
                  </div>
                  <div>
                    <strong>Source</strong>
                    <span>{sourceName}</span>
                  </div>
                  <div>
                    <strong>Nodes</strong>
                    <span>{flow.nodes.length}</span>
                  </div>
                  <div>
                    <strong>Edges</strong>
                    <span>{flow.edges.length}</span>
                  </div>
                </div>
              </Panel>
              {flow.coreLinkLegend.length > 0 && (
                <Panel position="top-right">
                  <div className="link-legend">
                    <h3>Core Links</h3>
                    {flow.coreLinkLegend.map((entry) => (
                      <button
                        type="button"
                        className={`link-legend-item${
                          selectedLegendLinkName === entry.name ? ' link-legend-item--active' : ''
                        }`}
                        key={entry.name}
                        onClick={() => setSelectedLegendLinkName(entry.name)}
                      >
                        <span className="link-legend-swatch" style={{ background: entry.color }} />
                        <span>{entry.name}</span>
                      </button>
                    ))}
                    {selectedLegendLink && (
                      <div className="link-legend-details">
                        <strong>{selectedLegendLink.name}</strong>
                        <div>
                          <span>bandwidth</span>
                          <code>{selectedLegendLink.bandwidth}</code>
                        </div>
                      </div>
                    )}
                  </div>
                </Panel>
              )}
              {flow.nodes.length === 0 && (
                <Panel position="top-center">
                  <div className="meta-panel">
                    <div>
                      <strong>Status</strong>
                      <span>No nodes to render</span>
                    </div>
                  </div>
                </Panel>
              )}
            </ReactFlow>
          ) : (
            <div className="graph-placeholder">
              <div className="graph-placeholder-content">
                <span className="graph-placeholder-icon">&#x2B22;</span>
                <h3>Select an architecture</h3>
                <p>Click on a node in the hierarchy tree to view its graph.</p>
                {parsed.error && <p className="error-line">{parsed.error}</p>}
              </div>
            </div>
          )}
        </section>

        {isEditorVisible && (
          <section className="editor-panel">
            <h2>Graph JSON</h2>
            <textarea
              value={jsonText}
              onChange={(event) => {
                setJsonText(event.target.value);
                setSourceName('editor');
                setLoadError(null);
              }}
              spellCheck={false}
            />
            {(loadError || parsed.error) && <p className="error-line">{loadError ?? parsed.error}</p>}
          </section>
        )}
      </main>

      {selectedCore && activeGraph?.intra_core && (
        <IntraCorePanel
          coreX={selectedCore.x}
          coreY={selectedCore.y}
          intraCoreGraph={activeGraph.intra_core}
          onClose={() => setSelectedCore(null)}
          onMemoryClick={onMemoryClick}
        />
      )}

      {selectedMemory && (
        <MemoryDetailPanel
          name={selectedMemory.name}
          region={selectedMemory.region}
          onClose={() => setSelectedMemory(null)}
        />
      )}
    </div>
  );
}

export default function App() {
  return (
    <ReactFlowProvider>
      <AppInner />
    </ReactFlowProvider>
  );
}
