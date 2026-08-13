import {cp, mkdir, readdir, rm} from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import {fileURLToPath} from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const siteDir = path.resolve(scriptDir, '..');
const sourceDir = path.resolve(siteDir, '../docs/.lavish/architecture');
const outputArg = process.argv[2];

if (!outputArg) {
  throw new Error('Usage: copy-archify-diagrams.mjs <output-directory>');
}

const outputDir = path.resolve(siteDir, outputArg);
let entries;

try {
  entries = await readdir(sourceDir, {withFileTypes: true});
} catch (error) {
  if (error && error.code === 'ENOENT') {
    throw new Error(
      `Missing ${sourceDir}. Deliver the Archify JSON into docs/.lavish/architecture before building Docusaurus.`,
    );
  }
  throw error;
}

const diagrams = entries.filter(
  (entry) => entry.isFile() && entry.name.endsWith('.html'),
);

if (diagrams.length === 0) {
  throw new Error(
    `No delivered Archify HTML files found in ${sourceDir}. Run Archify delivery first.`,
  );
}

await rm(outputDir, {recursive: true, force: true});
await mkdir(outputDir, {recursive: true});

for (const diagram of diagrams) {
  await cp(path.join(sourceDir, diagram.name), path.join(outputDir, diagram.name));
}

console.log(
  `Copied ${diagrams.length} Archify diagram${diagrams.length === 1 ? '' : 's'} to ${outputDir}.`,
);
