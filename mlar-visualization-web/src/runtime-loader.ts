import { parseArchitectureGraph, type ArchitectureGraph } from './schema';

export interface LoadedGraph {
  source: string;
  text: string;
  graph: ArchitectureGraph;
}

export function parseGraphText(text: string, source = 'editor'): LoadedGraph {
  const payload = JSON.parse(text) as unknown;
  const graph = parseArchitectureGraph(payload);
  return { source, text, graph };
}

export async function loadGraphFromFile(file: File): Promise<LoadedGraph> {
  const text = await file.text();
  return parseGraphText(text, file.name);
}

export async function loadGraphFromUrl(url: string): Promise<LoadedGraph> {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Failed to load ${url} (${response.status})`);
  }

  const text = await response.text();
  return parseGraphText(text, url);
}
