import {
  parseArchitectureGraph,
  parseAnyArchPayload,
  type ArchitectureGraph,
  type AnyArchPayload,
} from './schema';

export interface LoadedGraph {
  source: string;
  text: string;
  graph: ArchitectureGraph;
}

export interface LoadedPayload {
  source: string;
  text: string;
  payload: AnyArchPayload;
}

export function parseGraphText(text: string, source = 'editor'): LoadedGraph {
  const payload = JSON.parse(text) as unknown;
  const graph = parseArchitectureGraph(payload);
  return { source, text, graph };
}

export function parseAnyText(text: string, source = 'editor'): LoadedPayload {
  const raw = JSON.parse(text) as unknown;
  const payload = parseAnyArchPayload(raw);
  return { source, text, payload };
}

export async function loadGraphFromFile(file: File): Promise<LoadedPayload> {
  const text = await file.text();
  return parseAnyText(text, file.name);
}

export async function loadGraphFromUrl(url: string): Promise<LoadedPayload> {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Failed to load ${url} (${response.status})`);
  }

  const text = await response.text();
  return parseAnyText(text, url);
}
