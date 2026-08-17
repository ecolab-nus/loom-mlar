import {mkdtemp, mkdir, readFile, readdir, rename, rm} from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const supportedTypes = new Set([
  'architecture',
  'workflow',
  'sequence',
  'dataflow',
  'lifecycle',
]);

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const siteDir = path.resolve(scriptDir, '..');
const repoRoot = path.resolve(siteDir, '..');
const docsDir = path.join(repoRoot, 'docs');
const staticDir = path.join(siteDir, 'static');
const outputDir = path.join(staticDir, 'diagrams');
const archifyCli = path.join(repoRoot, 'tools/archify/bin/archify.mjs');

const entries = (await readdir(docsDir, {withFileTypes: true}))
  .filter((entry) => entry.isFile() && entry.name.endsWith('.json'))
  .sort((left, right) => left.name.localeCompare(right.name));

if (entries.length === 0) {
  throw new Error(`No Archify JSON specifications found in ${docsDir}.`);
}

await mkdir(staticDir, {recursive: true});
const stagingDir = await mkdtemp(path.join(staticDir, '.diagrams-build-'));

function runArchify(args, sourceName) {
  const result = spawnSync(process.execPath, [archifyCli, ...args], {
    cwd: repoRoot,
    stdio: 'inherit',
  });

  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(`Archify failed for ${sourceName} with exit code ${result.status}.`);
  }
}

try {
  for (const entry of entries) {
    const input = path.join(docsDir, entry.name);
    const specification = JSON.parse(await readFile(input, 'utf8'));
    const type = specification.diagram_type;

    if (!supportedTypes.has(type)) {
      throw new Error(
        `${input} has unsupported or missing diagram_type '${String(type)}'.`,
      );
    }

    const output = path.join(stagingDir, entry.name.replace(/\.json$/, '.html'));
    const commonArgs = [
      type,
      input,
      '--quality',
      'showcase',
      '--repo-root',
      repoRoot,
      '--json',
    ];

    runArchify(['validate', ...commonArgs], entry.name);
    runArchify(
      ['deliver', type, input, output, ...commonArgs.slice(2)],
      entry.name,
    );
  }

  await rm(outputDir, {recursive: true, force: true});
  await rename(stagingDir, outputDir);
  console.log(`Built ${entries.length} Archify diagrams in ${outputDir}.`);
} catch (error) {
  await rm(stagingDir, {recursive: true, force: true});
  throw error;
}
