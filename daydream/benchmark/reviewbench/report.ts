// Render the supplied report without importing eval.ts (which executes its CLI).
// The Python caller verifies this source checkout's pinned identity first.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

async function main() {
  const [source, metrics] = process.argv.slice(2);
  if (!source || !metrics) throw new Error("Expected the pinned source and metrics file");
  const { printSummary } = await import(pathToFileURL(resolve(source, "scripts/eval/print-summary.ts")).href);
  printSummary(JSON.parse(readFileSync(metrics, "utf8")));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
