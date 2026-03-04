import { useEffect, useMemo, useState, type ChangeEvent } from 'react';
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
import { architectureToFlow, type ArchFlowNode } from './flow';
import { loadGraphFromFile, loadGraphFromUrl, parseGraphText } from './runtime-loader';

const nodeTypes: NodeTypes = { archNode: ArchNode };
const DEFAULT_GRAPH_URL = '/sample-graph.json';

function AppInner() {
  const [jsonText, setJsonText] = useState('');
  const [sourceUrl, setSourceUrl] = useState(DEFAULT_GRAPH_URL);
  const [sourceName, setSourceName] = useState('none');
  const [loadError, setLoadError] = useState<string | null>(null);

  const loadFromUrl = async (url: string) => {
    try {
      const loaded = await loadGraphFromUrl(url);
      setJsonText(loaded.text);
      setSourceName(loaded.source);
      setLoadError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load URL';
      setLoadError(message);
    }
  };

  useEffect(() => {
    void loadFromUrl(DEFAULT_GRAPH_URL);
  }, []);

  const parsed = useMemo(() => {
    if (!jsonText.trim()) {
      return { graph: null, error: loadError ?? 'No JSON loaded yet.' };
    }

    try {
      const loaded = parseGraphText(jsonText, sourceName);
      return { graph: loaded.graph, error: null as string | null };
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Invalid JSON payload';
      return { graph: null, error: message };
    }
  }, [jsonText, sourceName, loadError]);

  const flow = useMemo(() => {
    if (!parsed.graph) {
      return { nodes: [] as ArchFlowNode[], edges: [] as Edge[] };
    }
    return architectureToFlow(parsed.graph);
  }, [parsed.graph]);

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
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load file';
      setLoadError(message);
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>MLAR Graph Flow</h1>
          <p>Runtime JSON loader for React Flow (schema: mlar.arch-graph.v1)</p>
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
        </div>
      </header>

      <main className="app-main">
        <section className="canvas-panel">
          <ReactFlow<ArchFlowNode, Edge>
            nodes={flow.nodes}
            edges={flow.edges}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.15 }}
          >
            <Controls />
            <MiniMap pannable zoomable />
            <Background color="#dbe8ef" gap={22} size={1} />
            <Panel position="top-left">
              <div className="meta-panel">
                <div>
                  <strong>Architecture</strong>
                  <span>{parsed.graph?.architecture.name ?? 'invalid payload'}</span>
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
        </section>

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
      </main>
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
