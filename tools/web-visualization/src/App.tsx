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
  ArchitectureHierarchy,
  AnyArchPayload,
  GraphMemoryRegion,
} from './schema';

const nodeTypes: NodeTypes = {
  archNode: ArchNode,
  coreArchNode: CoreArchNode,
  coreGridNode: CoreGridNode,
};
const DEFAULT_GRAPH_URL = '/sample-hierarchy.json';

type ViewMode = 'graph' | 'hierarchy';

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

function AppInner() {
  const [jsonText, setJsonText] = useState('');
  const [sourceUrl, setSourceUrl] = useState(DEFAULT_GRAPH_URL);
  const [sourceName, setSourceName] = useState('none');
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isEditorVisible, setIsEditorVisible] = useState(false);
  const [selectedCore, setSelectedCore] = useState<{ x: number; y: number } | null>(null);
  const [selectedLegendLinkName, setSelectedLegendLinkName] = useState<string | null>(null);
  const [selectedMemory, setSelectedMemory] = useState<{
    name: string;
    region: GraphMemoryRegion;
  } | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('graph');

  const loadFromUrl = async (url: string) => {
    try {
      const loaded = await loadGraphFromUrl(url);
      setJsonText(loaded.text);
      setSourceName(loaded.source);
      setLoadError(null);
      if (loaded.payload.type === 'hierarchy') {
        setViewMode('hierarchy');
      } else {
        setViewMode('graph');
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load URL';
      setLoadError(message);
    }
  };

  useEffect(() => {
    void loadFromUrl(DEFAULT_GRAPH_URL);
  }, []);

  const parsed = useMemo((): {
    payload: AnyArchPayload | null;
    graph: ArchitectureGraph | null;
    hierarchy: ArchitectureHierarchy | null;
    error: string | null;
  } => {
    if (!jsonText.trim()) {
      return { payload: null, graph: null, hierarchy: null, error: loadError ?? 'No JSON loaded yet.' };
    }

    try {
      const loaded = parseAnyText(jsonText, sourceName);
      if (loaded.payload.type === 'graph') {
        return { payload: loaded.payload, graph: loaded.payload.data, hierarchy: null, error: null };
      }
      return { payload: loaded.payload, graph: null, hierarchy: loaded.payload.data, error: null };
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Invalid JSON payload';
      return { payload: null, graph: null, hierarchy: null, error: message };
    }
  }, [jsonText, sourceName, loadError]);

  const onCoreClick = useCallback((x: number, y: number) => {
    setSelectedCore({ x, y });
  }, []);

  const onMemoryClick = useCallback((name: string, region: GraphMemoryRegion) => {
    setSelectedMemory({ name, region });
  }, []);

  const flow = useMemo(() => {
    if (!parsed.graph) {
      return {
        nodes: [] as AnyFlowNode[],
        edges: [] as Edge[],
        coreLinkLegend: [],
      } satisfies FlowConversionResult;
    }
    return architectureToFlow(parsed.graph, onCoreClick, onMemoryClick);
  }, [parsed.graph, onCoreClick, onMemoryClick]);

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
      if (loaded.payload.type === 'hierarchy') {
        setViewMode('hierarchy');
      } else {
        setViewMode('graph');
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load file';
      setLoadError(message);
    }
  };

  const canShowGraph = parsed.graph !== null;
  const canShowHierarchy = parsed.hierarchy !== null;

  const archName =
    parsed.graph?.architecture.name ?? parsed.hierarchy?.root.name ?? 'invalid payload';

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
            placeholder="/sample-graph.json"
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

          <div className="view-mode-toggle">
            <button
              type="button"
              className={`view-mode-btn ${viewMode === 'graph' ? 'view-mode-btn--active' : ''}`}
              disabled={!canShowGraph}
              onClick={() => setViewMode('graph')}
            >
              Graph
            </button>
            <button
              type="button"
              className={`view-mode-btn ${viewMode === 'hierarchy' ? 'view-mode-btn--active' : ''}`}
              disabled={!canShowHierarchy}
              onClick={() => setViewMode('hierarchy')}
            >
              Hierarchy
            </button>
          </div>
        </div>
      </header>

      <main className={`app-main${isEditorVisible ? ' app-main--editor-open' : ''}`}>
        <section className="canvas-panel">
          {viewMode === 'hierarchy' && parsed.hierarchy ? (
            <HierarchyView hierarchy={parsed.hierarchy} />
          ) : (
            <ReactFlow<AnyFlowNode, Edge>
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

      {selectedCore && parsed.graph?.intra_core && (
        <IntraCorePanel
          coreX={selectedCore.x}
          coreY={selectedCore.y}
          intraCoreGraph={parsed.graph.intra_core}
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
