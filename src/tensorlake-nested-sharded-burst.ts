import { createWriteStream, mkdirSync, writeFileSync } from 'node:fs';
import { once } from 'node:events';
import { resolve } from 'node:path';
import { randomUUID } from 'node:crypto';
import { config as loadEnv } from 'dotenv';
import { compute, type SandboxInterface } from 'computesdk';
import { tensorlake } from '@computesdk/tensorlake';

loadEnv({ quiet: true });

type ChildStatus = 'success' | 'failed' | 'readiness_failed' | 'destroy_failed';

type ChildResult = {
  runId: string;
  shardIndex: number;
  childIndex: number;
  globalIndex: number;
  startedAt: string;
  completedAt: string;
  createMs: number;
  commandMs: number | null;
  ttiMs: number | null;
  status: ChildStatus;
  sandboxId: string | null;
  nodeVersion: string | null;
  errorClass: string | null;
  errorMessage: string | null;
};

type Distribution = {
  count: number;
  min: number;
  p10: number;
  p25: number;
  p50: number;
  p75: number;
  p90: number;
  p95: number;
  p99: number;
  max: number;
  mean: number;
};

const DEFAULT_SHARDS = 1000;
const DEFAULT_CHILDREN_PER_SHARD = 100;
const DEFAULT_SHARD_CREATE_CONCURRENCY = 100;
const DEFAULT_SHARD_COMMAND_TIMEOUT_MS = 300_000;
const DEFAULT_DESTROY_TIMEOUT_MS = 15_000;

const shardWorkerSource = String.raw`
const RUN_ID = process.env.RUN_ID;
const SHARD_INDEX = Number(process.env.SHARD_INDEX);
const CHILDREN_PER_SHARD = Number(process.env.CHILDREN_PER_SHARD);
const TENSORLAKE_API_KEY = process.env.TENSORLAKE_API_KEY;
const API_URL = process.env.TENSORLAKE_API_URL || 'https://api.tensorlake.ai';
const LIFECYCLE_URL = process.env.TENSORLAKE_SANDBOX_LIFECYCLE_URL || lifecycleUrlFromApiUrl(API_URL);
const PROXY_URL = process.env.TENSORLAKE_SANDBOX_PROXY_URL || 'https://sandbox.tensorlake.ai';
const REQUEST_TIMEOUT_MS = Number(process.env.REQUEST_TIMEOUT_MS || 120000);
const COMMAND_TIMEOUT_MS = Number(process.env.COMMAND_TIMEOUT_MS || 30000);

if (!RUN_ID || !Number.isInteger(SHARD_INDEX) || !Number.isInteger(CHILDREN_PER_SHARD) || !TENSORLAKE_API_KEY) {
  throw new Error('Missing shard worker environment');
}

function lifecycleUrlFromApiUrl(apiUrl) {
  try {
    const parsed = new URL(apiUrl);
    if (parsed.hostname.startsWith('api.')) {
      parsed.hostname = 'sandbox.' + parsed.hostname.slice(4);
      return parsed.toString().replace(/\/$/, '');
    }
  } catch {}
  return apiUrl;
}

function headers(extra = {}) {
  return {
    Authorization: 'Bearer ' + TENSORLAKE_API_KEY,
    'Content-Type': 'application/json',
    'X-Tensorlake-Request-Timeout-Ms': String(REQUEST_TIMEOUT_MS),
    ...extra,
  };
}

function classifyError(err) {
  const message = String(err?.message ?? err).toLowerCase();
  if (message.includes('timeout') || message.includes('timed out')) return 'timeout';
  if (message.includes('429')) return 'rate_limit';
  if (message.includes('quota')) return 'quota';
  if (err?.code) return String(err.code);
  return 'error';
}

function shortError(err) {
  const message = err instanceof Error ? err.message : String(err);
  return message.length > 500 ? message.slice(0, 500) : message;
}

async function fetchJson(url, init, timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    const text = await response.text();
    const data = text ? JSON.parse(text) : null;
    if (!response.ok) {
      throw new Error('HTTP ' + response.status + ': ' + text);
    }
    return data;
  } finally {
    clearTimeout(timer);
  }
}

async function createSandbox() {
  const raw = await fetchJson(LIFECYCLE_URL + '/sandboxes', {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({ resources: { cpus: 1, memory_mb: 1024 } }),
  });
  return {
    sandboxId: raw.sandbox_id || raw.sandboxId,
    status: raw.status,
    routingHint: raw.routing_hint || raw.routingHint,
  };
}

async function waitForRunning(sandboxId) {
  const deadline = Date.now() + REQUEST_TIMEOUT_MS;
  let last;
  while (Date.now() < deadline) {
    const raw = await fetchJson(LIFECYCLE_URL + '/sandboxes/' + encodeURIComponent(sandboxId), {
      method: 'GET',
      headers: headers(),
    });
    last = raw;
    const status = raw.status;
    if (status === 'running') {
      return {
        sandboxId: raw.sandbox_id || raw.sandboxId || sandboxId,
        routingHint: raw.routing_hint || raw.routingHint,
      };
    }
    if (status === 'terminated' || status === 'suspended' || status === 'timeout') {
      throw new Error('Sandbox ' + sandboxId + ' reached status ' + status);
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error('Sandbox ' + sandboxId + ' did not become running; last=' + JSON.stringify(last));
}

async function deleteSandbox(sandboxId) {
  await fetch(LIFECYCLE_URL + '/sandboxes/' + encodeURIComponent(sandboxId), {
    method: 'DELETE',
    headers: headers(),
  }).catch(() => {});
}

async function runNodeVersion(sandboxId, routingHint) {
  const response = await fetch(PROXY_URL + '/api/v1/processes/run', {
    method: 'POST',
    headers: headers({
      'X-Tensorlake-Sandbox-Id': sandboxId,
      ...(routingHint ? { 'X-Tensorlake-Route-Hint': routingHint } : {}),
    }),
    body: JSON.stringify({ command: 'sh', args: ['-c', 'node -v'] }),
  });
  if (!response.ok) {
    throw new Error('run HTTP ' + response.status + ': ' + await response.text());
  }
  return parseSseProcessResult(response);
}

async function parseSseProcessResult(response) {
  const decoder = new TextDecoder();
  let buffer = '';
  let stdout = [];
  let stderr = [];
  let exitCode = -1;

  for await (const chunk of response.body) {
    buffer += decoder.decode(chunk, { stream: true });
    let index;
    while ((index = buffer.indexOf('\n\n')) >= 0) {
      const event = buffer.slice(0, index);
      buffer = buffer.slice(index + 2);
      for (const line of event.split('\n')) {
        if (!line.startsWith('data:')) continue;
        const raw = line.slice(5).trim();
        if (!raw) continue;
        const data = JSON.parse(raw);
        if (typeof data.line === 'string') {
          if (data.stream === 'stderr') stderr.push(data.line);
          else stdout.push(data.line);
        } else if (typeof data.exit_code === 'number') {
          exitCode = data.exit_code;
        } else if (typeof data.signal === 'number') {
          exitCode = -data.signal;
        }
      }
    }
  }

  return { exitCode, stdout: stdout.join('\n'), stderr: stderr.join('\n') };
}

async function runChild(childIndex) {
  const globalIndex = SHARD_INDEX * CHILDREN_PER_SHARD + childIndex;
  const startedAt = new Date().toISOString();
  const t0 = performance.now();
  let sandboxId = null;
  let routingHint = null;
  let createMs = 0;
  let commandMs = null;
  const result = {
    runId: RUN_ID,
    shardIndex: SHARD_INDEX,
    childIndex,
    globalIndex,
    startedAt,
    completedAt: '',
    createMs: 0,
    commandMs: null,
    ttiMs: null,
    status: 'success',
    sandboxId: null,
    nodeVersion: null,
    errorClass: null,
    errorMessage: null,
  };

  try {
    const created = await createSandbox();
    sandboxId = created.sandboxId;
    routingHint = created.routingHint;
    const running = created.status === 'running' ? created : await waitForRunning(sandboxId);
    routingHint = running.routingHint || routingHint;
    createMs = Math.round(performance.now() - t0);
    result.createMs = createMs;
    result.sandboxId = sandboxId;

    const commandStart = performance.now();
    const command = await runNodeVersion(sandboxId, routingHint);
    commandMs = Math.round(performance.now() - commandStart);
    result.commandMs = commandMs;
    result.ttiMs = createMs + commandMs;
    result.nodeVersion = command.stdout.trim();
    if (command.exitCode !== 0) {
      result.status = 'readiness_failed';
      result.errorClass = 'non_zero_exit';
      result.errorMessage = command.stderr || 'node -v exited with ' + command.exitCode;
    }
  } catch (err) {
    result.status = sandboxId ? 'readiness_failed' : 'failed';
    result.createMs = sandboxId ? createMs : Math.round(performance.now() - t0);
    result.commandMs = commandMs;
    result.ttiMs = createMs && commandMs !== null ? createMs + commandMs : null;
    result.sandboxId = sandboxId;
    result.errorClass = classifyError(err);
    result.errorMessage = shortError(err);
  } finally {
    if (sandboxId) {
      try {
        await deleteSandbox(sandboxId);
      } catch (err) {
        if (result.status === 'success') {
          result.status = 'destroy_failed';
          result.errorClass = classifyError(err);
          result.errorMessage = shortError(err);
        }
      }
    }
    result.completedAt = new Date().toISOString();
  }

  return result;
}

const shardStartedAt = new Date().toISOString();
const wallStart = performance.now();
const results = await Promise.all(Array.from({ length: CHILDREN_PER_SHARD }, (_, i) => runChild(i)));
for (const result of results) {
  console.log(JSON.stringify({ type: 'child-result', result }));
}
console.log(JSON.stringify({
  type: 'shard-result',
  runId: RUN_ID,
  shardIndex: SHARD_INDEX,
  startedAt: shardStartedAt,
  completedAt: new Date().toISOString(),
  wallClockMs: Math.round(performance.now() - wallStart),
  resultCount: results.length,
}));
`;

function usage(exitCode = 1): never {
  console.error(`Usage:
  npm run bench:tensorlake:nested

Options:
  --shards <n>                       Shard sandboxes to create first. Default: ${DEFAULT_SHARDS}
  --children-per-shard <n>           Child sandboxes each shard creates in parallel. Default: ${DEFAULT_CHILDREN_PER_SHARD}
  --shard-create-concurrency <n>     Local concurrency for creating shard sandboxes. Default: ${DEFAULT_SHARD_CREATE_CONCURRENCY}
  --shard-command-timeout-ms <n>     Timeout for each shard worker command. Default: ${DEFAULT_SHARD_COMMAND_TIMEOUT_MS}
  --destroy-timeout-ms <n>           Timeout for shard cleanup. Default: ${DEFAULT_DESTROY_TIMEOUT_MS}
  --output-dir <path>                Output directory. Default: results/<run-id>

Environment:
  TENSORLAKE_API_KEY is required.
`);
  process.exit(exitCode);
}

function parsePositiveInt(value: string, flag: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`${flag} must be a positive integer`);
  return parsed;
}

function parseArgs(argv: string[]) {
  const parsed = {
    shardCount: DEFAULT_SHARDS,
    childrenPerShard: DEFAULT_CHILDREN_PER_SHARD,
    shardCreateConcurrency: DEFAULT_SHARD_CREATE_CONCURRENCY,
    shardCommandTimeoutMs: DEFAULT_SHARD_COMMAND_TIMEOUT_MS,
    destroyTimeoutMs: DEFAULT_DESTROY_TIMEOUT_MS,
    outputDir: '',
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const value = argv[i + 1];
    if (arg === '--help' || arg === '-h') usage(0);
    if (!arg.startsWith('--') || !value || value.startsWith('--')) usage();
    i++;
    switch (arg) {
      case '--shards':
        parsed.shardCount = parsePositiveInt(value, arg);
        break;
      case '--children-per-shard':
      case '--sandboxes-per-shard':
        parsed.childrenPerShard = parsePositiveInt(value, arg);
        break;
      case '--shard-create-concurrency':
        parsed.shardCreateConcurrency = parsePositiveInt(value, arg);
        break;
      case '--shard-command-timeout-ms':
        parsed.shardCommandTimeoutMs = parsePositiveInt(value, arg);
        break;
      case '--destroy-timeout-ms':
        parsed.destroyTimeoutMs = parsePositiveInt(value, arg);
        break;
      case '--output-dir':
        parsed.outputDir = value;
        break;
      default:
        usage();
    }
  }
  return parsed;
}

function distribution(values: number[]): Distribution | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const at = (q: number) => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * q))];
  return {
    count: sorted.length,
    min: sorted[0],
    p10: at(0.10),
    p25: at(0.25),
    p50: at(0.50),
    p75: at(0.75),
    p90: at(0.90),
    p95: at(0.95),
    p99: at(0.99),
    max: sorted[sorted.length - 1],
    mean: Math.round(sorted.reduce((sum, value) => sum + value, 0) / sorted.length),
  };
}

async function withTimeout<T>(promise: Promise<T>, ms: number, message: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<T>((_, reject) => {
        timer = setTimeout(() => reject(new Error(message)), ms);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

async function mapLimit<T, R>(items: T[], concurrency: number, fn: (item: T, index: number) => Promise<R>): Promise<R[]> {
  const results = new Array<R>(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (true) {
      const index = next++;
      if (index >= items.length) return;
      results[index] = await fn(items[index], index);
    }
  }));
  return results;
}

async function writeJsonLine(stream: NodeJS.WritableStream, value: unknown): Promise<void> {
  if (!stream.write(`${JSON.stringify(value)}\n`)) await once(stream, 'drain');
}

function shellSingleQuote(value: string): string {
  return `'${value.replaceAll("'", "'\\''")}'`;
}

function shardCommand(runId: string, shardIndex: number, childrenPerShard: number, apiKey: string): string {
  return [
    'set -eu',
    'mkdir -p /tmp/tensorlake-nested-bench',
    'cd /tmp/tensorlake-nested-bench',
    `cat > worker.mjs <<'JS'\n${shardWorkerSource}\nJS`,
    [
      `RUN_ID=${shellSingleQuote(runId)}`,
      `SHARD_INDEX=${shardIndex}`,
      `CHILDREN_PER_SHARD=${childrenPerShard}`,
      `TENSORLAKE_API_KEY=${shellSingleQuote(apiKey)}`,
      'node worker.mjs',
    ].join(' '),
  ].join('\n');
}

function parseShardOutput(stdout: string): { results: ChildResult[]; wallClockMs: number } {
  const lines = stdout.trim().split('\n').filter(Boolean);
  const results: ChildResult[] = [];
  let wallClockMs = 0;

  for (const line of lines) {
    try {
      const parsed = JSON.parse(line) as {
        type?: string;
        result?: ChildResult;
        results?: ChildResult[];
        wallClockMs?: number;
      };
      if (parsed.type === 'child-result' && parsed.result) {
        results.push(parsed.result);
      } else if (parsed.type === 'shard-result') {
        wallClockMs = parsed.wallClockMs ?? wallClockMs;
        if (Array.isArray(parsed.results)) results.push(...parsed.results);
      }
    } catch {
      // Ignore non-JSON output.
    }
  }

  if (results.length > 0) return { results, wallClockMs };

  for (let i = lines.length - 1; i >= 0; i--) {
    try {
      const parsed = JSON.parse(lines[i]) as { type?: string; results?: ChildResult[]; wallClockMs?: number };
      if (parsed.type === 'shard-result' && Array.isArray(parsed.results)) {
        return { results: parsed.results, wallClockMs: parsed.wallClockMs ?? 0 };
      }
    } catch {
      // Ignore non-JSON output.
    }
  }
  throw new Error(`Shard did not emit a parseable result. Last output: ${stdout.slice(-1000)}`);
}

async function destroyShard(sandbox: SandboxInterface, timeoutMs: number): Promise<void> {
  await withTimeout(sandbox.destroy(), timeoutMs, `Shard destroy timed out after ${timeoutMs}ms`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const apiKey = process.env.TENSORLAKE_API_KEY;
  if (!apiKey) throw new Error('Missing TENSORLAKE_API_KEY');

  const runId = randomUUID();
  const outputDir = resolve(args.outputDir || `results/${runId}`);
  mkdirSync(outputDir, { recursive: true });
  const rawJsonl = resolve(outputDir, 'raw.jsonl');
  const shardJsonl = resolve(outputDir, 'shards.jsonl');
  const summaryJson = resolve(outputDir, 'summary.json');
  const rawStream = createWriteStream(rawJsonl, { flags: 'wx' });
  const shardStream = createWriteStream(shardJsonl, { flags: 'wx' });
  rawStream.setMaxListeners(0);
  shardStream.setMaxListeners(0);

  const sdk = compute({ provider: tensorlake({ apiKey }) });
  const totalChildren = args.shardCount * args.childrenPerShard;
  const startedAt = new Date().toISOString();

  console.log(`run_id=${runId}`);
  console.log(`provider=tensorlake topology=nested`);
  console.log(`shards=${args.shardCount} children_per_shard=${args.childrenPerShard} total_children=${totalChildren}`);
  console.log(`output=${outputDir}`);
  console.log(`creating shard sandboxes with concurrency=${args.shardCreateConcurrency}...`);

  let createdShards = 0;
  const shardCreateStart = performance.now();
  const shardSandboxes = await mapLimit(
    Array.from({ length: args.shardCount }, (_, i) => i),
    args.shardCreateConcurrency,
    async (shardIndex) => {
      const t0 = performance.now();
      const sandbox = await sdk.sandbox.create();
      createdShards++;
      if (createdShards % 25 === 0 || createdShards === args.shardCount) {
        console.log(`created_shards=${createdShards}/${args.shardCount}`);
      }
      return { shardIndex, sandbox, createMs: Math.round(performance.now() - t0) };
    },
  );

  console.log(`all shard sandboxes created in ${Math.round(performance.now() - shardCreateStart)}ms; releasing workers`);

  const workerStart = performance.now();
  let shardsDone = 0;
  let childResults = 0;
  const progress = setInterval(() => {
    const elapsed = ((performance.now() - workerStart) / 1000).toFixed(1);
    console.log(`elapsed=${elapsed}s shards_done=${shardsDone}/${args.shardCount} child_results=${childResults}/${totalChildren}`);
  }, 30_000);

  const shardSummaries = await Promise.all(shardSandboxes.map(async ({ shardIndex, sandbox, createMs }) => {
    let results: ChildResult[] = [];
    let workerWallClockMs = 0;
    let status: 'success' | 'failed' = 'success';
    let errorMessage: string | null = null;
    const shardStartedAt = new Date().toISOString();

    try {
      const commandResult = await withTimeout(
        sandbox.runCommand(shardCommand(runId, shardIndex, args.childrenPerShard, apiKey)),
        args.shardCommandTimeoutMs,
        `Shard worker timed out after ${args.shardCommandTimeoutMs}ms`,
      );
      if (commandResult.exitCode !== 0) {
        throw new Error(commandResult.stderr || `Shard worker exited ${commandResult.exitCode}`);
      }
      const parsed = parseShardOutput(commandResult.stdout);
      results = parsed.results;
      workerWallClockMs = parsed.wallClockMs;
      for (const result of results) await writeJsonLine(rawStream, result);
      childResults += results.length;
    } catch (err) {
      status = 'failed';
      errorMessage = err instanceof Error ? err.message : String(err);
    } finally {
      try {
        await destroyShard(sandbox, args.destroyTimeoutMs);
      } catch {
        // Cleanup failure is visible through Tensorlake list; do not hide benchmark data.
      }
    }

    shardsDone++;
    const succeeded = results.filter((result) => result.status === 'success').length;
    const summary = {
      runId,
      shardIndex,
      shardSandboxId: sandbox.sandboxId,
      shardCreateMs: createMs,
      startedAt: shardStartedAt,
      completedAt: new Date().toISOString(),
      workerWallClockMs,
      attempted: results.length,
      succeeded,
      status,
      errorMessage,
    };
    await writeJsonLine(shardStream, summary);
    console.log(`shard ${shardIndex} ${status} succeeded=${succeeded}/${args.childrenPerShard}`);
    return summary;
  }));

  clearInterval(progress);
  await new Promise<void>((resolveDone) => rawStream.end(() => resolveDone()));
  await new Promise<void>((resolveDone) => shardStream.end(() => resolveDone()));

  const allResults = await import('node:fs/promises')
    .then((fs) => fs.readFile(rawJsonl, 'utf-8'))
    .then((contents) => contents.trim() ? contents.trim().split('\n').map((line) => JSON.parse(line) as ChildResult) : []);
  const successes = allResults.filter((result) => result.status === 'success');
  const childStatusCounts = allResults.reduce<Record<string, number>>((acc, result) => {
    acc[result.status] = (acc[result.status] ?? 0) + 1;
    return acc;
  }, {});

  const summary = {
    runId,
    provider: 'tensorlake',
    topology: 'nested-sharded',
    shardCount: args.shardCount,
    childrenPerShard: args.childrenPerShard,
    totalChildren,
    startedAt,
    completedAt: new Date().toISOString(),
    wallClockMs: Math.round(performance.now() - workerStart),
    shardCreateMs: distribution(shardSandboxes.map((shard) => shard.createMs)),
    shardStatusCounts: shardSummaries.reduce<Record<string, number>>((acc, shard) => {
      acc[shard.status] = (acc[shard.status] ?? 0) + 1;
      return acc;
    }, {}),
    childStatusCounts,
    createMs: distribution(successes.map((result) => result.createMs)),
    commandMs: distribution(successes.flatMap((result) => result.commandMs === null ? [] : [result.commandMs])),
    ttiMs: distribution(successes.flatMap((result) => result.ttiMs === null ? [] : [result.ttiMs])),
    files: { rawJsonl, shardJsonl, summaryJson },
  };

  writeFileSync(summaryJson, `${JSON.stringify(summary, null, 2)}\n`);
  console.log(`complete child_success=${childStatusCounts.success ?? 0}/${totalChildren} wall_clock_ms=${summary.wallClockMs}`);
  console.log(`summary=${summaryJson}`);
  console.log(`raw=${rawJsonl}`);
  console.log(`shards=${shardJsonl}`);
}

main().catch((err) => {
  console.error(err instanceof Error ? err.stack || err.message : err);
  process.exit(1);
});
