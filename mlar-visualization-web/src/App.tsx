import { useMemo, useState, type ChangeEvent } from 'react';
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
import { parseArchitectureGraph, type ArchitectureGraph } from './schema';
import sampleGraph from './sample-graph.json';

const nodeTypes: NodeTypes = { archNode: ArchNode };

function parseInput(text: string): ArchitectureGraph {
  return parseArchitectureGraph(JSON.parse(text) as unknown);
}

function AppInner() {
  const initialText = JSON.stringify(sampleGraph, null, 2);
  const [jsonText, setJsonText] = useState(initialText);

  const parsed = useMemo(() => {
    try {
      const graph = parseInput(jsonText);
      return { graph, error: null as string | null };
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Invalid JSON payload';
      return { graph: null, error: message };
    }
  }, [jsonText]);

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

    const text = await file.text();
    setJsonText(text);
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>MLAR Graph Flow</h1>
          <p>React Flow prototype for MLAR JSON schema ({sampleGraph.schema_version})</p>
        </div>
        <label className="upload-button">
          Load JSON
          <input type="file" accept="application/json" onChange={onUpload} />
        </label>
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
            onChange={(event) => setJsonText(event.target.value)}
            spellCheck={false}
          />
          {parsed.error && <p className="error-line">{parsed.error}</p>}
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
