import { createWriteStream, mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { once } from 'node:events';
import { randomUUID } from 'node:crypto';
import { config as loadEnv } from 'dotenv';
import { compute, type SandboxInterface } from 'computesdk';
import { tensorlake } from '@computesdk/tensorlake';

loadEnv({ quiet: true });

type Status = 'success' | 'failed' | 'readiness_failed' | 'destroy_failed';

type SandboxResult = {
  runId: string;
  shardIndex: number;
  sandboxIndex: number;
  globalIndex: number;
  startedAt: string;
  completedAt: string;
  createMs: number;
  commandMs: number | null;
  ttiMs: number | null;
  status: Status;
  sandboxId: string | null;
  nodeVersion: string | null;
  errorClass: string | null;
  errorMessage: string | null;
};

type Summary = {
  runId: string;
  provider: 'tensorlake';
  shardCount: number;
  sandboxesPerShard: number;
  totalSandboxes: number;
  sandboxOptions: Record<string, unknown>;
  startedAt: string;
  completedAt: string;
  wallClockMs: number;
  statusCounts: Record<Status, number>;
  createMs: Distribution | null;
  commandMs: Distribution | null;
  ttiMs: Distribution | null;
  byShard: Array<{
    shardIndex: number;
    attempted: number;
    succeeded: number;
    wallClockMs: number;
    createMs: Distribution | null;
    commandMs: Distribution | null;
    ttiMs: Distribution | null;
  }>;
  files: {
    rawJsonl: string;
    summaryJson: string;
  };
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
const DEFAULT_SANDBOXES_PER_SHARD = 100;
const DEFAULT_REQUEST_TIMEOUT_MS = 120_000;
const DEFAULT_COMMAND_TIMEOUT_MS = 30_000;
const DEFAULT_DESTROY_TIMEOUT_MS = 15_000;
function usage(exitCode = 1): never {
  console.error(`Usage:
  npm run bench:tensorlake

Options:
  --shards <n>                 Number of shard tasks to create first. Default: ${DEFAULT_SHARDS}
  --sandboxes-per-shard <n>    Sandboxes each shard bursts in parallel. Default: ${DEFAULT_SANDBOXES_PER_SHARD}
  --request-timeout-ms <n>     Timeout for sandbox creation. Default: ${DEFAULT_REQUEST_TIMEOUT_MS}
  --command-timeout-ms <n>     Timeout for node -v. Default: ${DEFAULT_COMMAND_TIMEOUT_MS}
  --destroy-timeout-ms <n>     Timeout for cleanup. Default: ${DEFAULT_DESTROY_TIMEOUT_MS}
  --output-dir <path>          Directory for raw JSONL and summary JSON. Default: results/<run-id>
  --image <image>              Tensorlake image/template override. Not passed by default.
  --timeout-ms <n>             Tensorlake sandbox timeout option. Not passed by default.
  --cpus <n>                   Tensorlake cpus option. Not passed by default.
  --memory-mb <n>              Tensorlake memoryMb option. Not passed by default.
  --ephemeral-disk-mb <n>      Tensorlake ephemeralDiskMb option. Not passed by default.

Environment:
  TENSORLAKE_API_KEY is required.
`);
  process.exit(exitCode);
}

function parseArgs(argv: string[]) {
  const parsed = {
    shardCount: DEFAULT_SHARDS,
    sandboxesPerShard: DEFAULT_SANDBOXES_PER_SHARD,
    requestTimeoutMs: DEFAULT_REQUEST_TIMEOUT_MS,
    commandTimeoutMs: DEFAULT_COMMAND_TIMEOUT_MS,
    destroyTimeoutMs: DEFAULT_DESTROY_TIMEOUT_MS,
    outputDir: '',
    image: undefined as string | undefined,
    timeoutMs: undefined as number | undefined,
    cpus: undefined as number | undefined,
    memoryMb: undefined as number | undefined,
    ephemeralDiskMb: undefined as number | undefined,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const value = argv[i + 1];

    if (arg === '--help' || arg === '-h') usage(0);
    if (!arg.startsWith('--')) usage();
    if (!value || value.startsWith('--')) usage();

    i++;
    switch (arg) {
      case '--shards':
        parsed.shardCount = parsePositiveInt(value, arg);
        break;
      case '--sandboxes-per-shard':
        parsed.sandboxesPerShard = parsePositiveInt(value, arg);
        break;
      case '--request-timeout-ms':
        parsed.requestTimeoutMs = parsePositiveInt(value, arg);
        break;
      case '--command-timeout-ms':
        parsed.commandTimeoutMs = parsePositiveInt(value, arg);
        break;
      case '--destroy-timeout-ms':
        parsed.destroyTimeoutMs = parsePositiveInt(value, arg);
        break;
      case '--output-dir':
        parsed.outputDir = value;
        break;
      case '--image':
        parsed.image = value;
        break;
      case '--timeout-ms':
        parsed.timeoutMs = parsePositiveInt(value, arg);
        break;
      case '--cpus':
        parsed.cpus = parsePositiveNumber(value, arg);
        break;
      case '--memory-mb':
        parsed.memoryMb = parsePositiveInt(value, arg);
        break;
      case '--ephemeral-disk-mb':
        parsed.ephemeralDiskMb = parsePositiveInt(value, arg);
        break;
      default:
        usage();
    }
  }

  return parsed;
}

function parsePositiveInt(value: string, flag: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${flag} must be a positive integer, got "${value}"`);
  }
  return parsed;
}

function parsePositiveNumber(value: string, flag: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${flag} must be a positive number, got "${value}"`);
  }
  return parsed;
}

function distribution(values: number[]): Distribution | null {
  if (values.length === 0) return null;

  const sorted = [...values].sort((a, b) => a - b);
  const at = (q: number) => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * q))];
  const mean = sorted.reduce((sum, value) => sum + value, 0) / sorted.length;

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
    mean: Math.round(mean),
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

function classifyError(err: unknown): string {
  const error = err as { message?: string; status?: number; statusCode?: number; code?: string };
  const message = String(error?.message ?? err).toLowerCase();
  if (message.includes('timeout') || message.includes('timed out')) return 'timeout';
  if (typeof error?.status === 'number' || typeof error?.statusCode === 'number') return 'http_error';
  if (error?.code) return String(error.code);
  return 'error';
}

function shortError(err: unknown): string {
  const message = err instanceof Error ? err.message : String(err);
  return message.length > 500 ? message.slice(0, 500) : message;
}

async function writeJsonLine(stream: NodeJS.WritableStream, result: SandboxResult): Promise<void> {
  if (!stream.write(`${JSON.stringify(result)}\n`)) {
    await once(stream, 'drain');
  }
}

async function destroySandbox(sandbox: SandboxInterface, timeoutMs: number): Promise<void> {
  await withTimeout(sandbox.destroy(), timeoutMs, `Sandbox destroy timed out after ${timeoutMs}ms`);
}

function createGate() {
  let release!: () => void;
  const promise = new Promise<void>((resolve) => {
    release = resolve;
  });

  return { promise, release };
}

async function runSandbox(
  sdk: ReturnType<typeof compute>,
  runId: string,
  shardIndex: number,
  sandboxIndex: number,
  sandboxesPerShard: number,
  sandboxOptions: Record<string, unknown> | undefined,
  requestTimeoutMs: number,
  commandTimeoutMs: number,
  destroyTimeoutMs: number,
  stream: NodeJS.WritableStream,
): Promise<SandboxResult> {
  const globalIndex = shardIndex * sandboxesPerShard + sandboxIndex;
  const startedAt = new Date().toISOString();
  const t0 = performance.now();
  let sandbox: SandboxInterface | null = null;
  let createMs = 0;
  let commandMs: number | null = null;

  const result: SandboxResult = {
    runId,
    shardIndex,
    sandboxIndex,
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
    sandbox = await withTimeout(
      sdk.sandbox.create(sandboxOptions),
      requestTimeoutMs,
      `Sandbox creation timed out after ${requestTimeoutMs}ms`,
    );
    createMs = Math.round(performance.now() - t0);
    result.createMs = createMs;
    result.sandboxId = sandbox.sandboxId;

    const commandStart = performance.now();
    const commandResult = await withTimeout(
      sandbox.runCommand('node -v', { timeout: commandTimeoutMs }),
      commandTimeoutMs,
      `node -v timed out after ${commandTimeoutMs}ms`,
    );
    commandMs = Math.round(performance.now() - commandStart);
    result.commandMs = commandMs;
    result.ttiMs = createMs + commandMs;
    result.nodeVersion = commandResult.stdout.trim();

    if (commandResult.exitCode !== 0) {
      result.status = 'readiness_failed';
      result.errorClass = 'non_zero_exit';
      result.errorMessage = commandResult.stderr || `node -v exited with ${commandResult.exitCode}`;
    }
  } catch (err) {
    if (sandbox) {
      result.status = 'readiness_failed';
      result.createMs = createMs;
      result.commandMs = commandMs;
      result.ttiMs = createMs && commandMs !== null ? createMs + commandMs : null;
    } else {
      result.status = 'failed';
      result.createMs = Math.round(performance.now() - t0);
    }
    result.errorClass = classifyError(err);
    result.errorMessage = shortError(err);
  } finally {
    if (sandbox) {
      try {
        await destroySandbox(sandbox, destroyTimeoutMs);
      } catch (err) {
        if (result.status === 'success') {
          result.status = 'destroy_failed';
          result.errorClass = classifyError(err);
          result.errorMessage = shortError(err);
        }
      }
    }

    result.completedAt = new Date().toISOString();
    await writeJsonLine(stream, result);
  }

  return result;
}

async function runShard(
  sdk: ReturnType<typeof compute>,
  runId: string,
  shardIndex: number,
  sandboxesPerShard: number,
  sandboxOptions: Record<string, unknown> | undefined,
  requestTimeoutMs: number,
  commandTimeoutMs: number,
  destroyTimeoutMs: number,
  gate: Promise<void>,
  stream: NodeJS.WritableStream,
) {
  await gate;
  const shardStart = performance.now();
  const results = await Promise.all(
    Array.from({ length: sandboxesPerShard }, (_, sandboxIndex) =>
      runSandbox(
        sdk,
        runId,
        shardIndex,
        sandboxIndex,
        sandboxesPerShard,
        sandboxOptions,
        requestTimeoutMs,
        commandTimeoutMs,
        destroyTimeoutMs,
        stream,
      ),
    ),
  );

  const successes = results.filter((result) => result.status === 'success');
  return {
    shardIndex,
    attempted: results.length,
    succeeded: successes.length,
    wallClockMs: Math.round(performance.now() - shardStart),
    createMs: distribution(successes.map((result) => result.createMs)),
    commandMs: distribution(successes.flatMap((result) => result.commandMs === null ? [] : [result.commandMs])),
    ttiMs: distribution(successes.flatMap((result) => result.ttiMs === null ? [] : [result.ttiMs])),
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!process.env.TENSORLAKE_API_KEY) {
    throw new Error('Missing TENSORLAKE_API_KEY');
  }

  const runId = randomUUID();
  const outputDir = resolve(args.outputDir || `results/${runId}`);
  mkdirSync(outputDir, { recursive: true });

  const rawJsonl = resolve(outputDir, 'raw.jsonl');
  const summaryJson = resolve(outputDir, 'summary.json');
  const stream = createWriteStream(rawJsonl, { flags: 'wx' });
  stream.setMaxListeners(0);

  const configuredSandboxOptions: Record<string, unknown> = {};
  if (args.image) configuredSandboxOptions.image = args.image;
  if (args.timeoutMs !== undefined) configuredSandboxOptions.timeout = args.timeoutMs;
  if (args.cpus !== undefined) configuredSandboxOptions.cpus = args.cpus;
  if (args.memoryMb !== undefined) configuredSandboxOptions.memoryMb = args.memoryMb;
  if (args.ephemeralDiskMb !== undefined) configuredSandboxOptions.ephemeralDiskMb = args.ephemeralDiskMb;
  const sandboxOptions = Object.keys(configuredSandboxOptions).length > 0 ? configuredSandboxOptions : undefined;

  const sdk = compute({
    provider: tensorlake({
      apiKey: process.env.TENSORLAKE_API_KEY,
      apiUrl: process.env.TENSORLAKE_API_URL,
      proxyUrl: process.env.TENSORLAKE_SANDBOX_PROXY_URL,
    }),
  });

  const totalSandboxes = args.shardCount * args.sandboxesPerShard;
  const gate = createGate();
  const startedAt = new Date().toISOString();

  console.log(`run_id=${runId}`);
  console.log(`provider=tensorlake`);
  console.log(`shards=${args.shardCount} sandboxes_per_shard=${args.sandboxesPerShard} total=${totalSandboxes}`);
  console.log(`output=${outputDir}`);
  console.log('creating shard tasks before burst...');

  let shardsDone = 0;
  const shardPromises = Array.from({ length: args.shardCount }, (_, shardIndex) =>
    runShard(
      sdk,
      runId,
      shardIndex,
      args.sandboxesPerShard,
      sandboxOptions,
      args.requestTimeoutMs,
      args.commandTimeoutMs,
      args.destroyTimeoutMs,
      gate.promise,
      stream,
    ).finally(() => {
      shardsDone++;
    }),
  );

  console.log(`all ${args.shardCount} shard tasks created; releasing burst`);
  const wallStart = performance.now();
  gate.release();

  const progress = setInterval(() => {
    const elapsed = (performance.now() - wallStart) / 1000;
    console.log(`elapsed=${elapsed.toFixed(1)}s shards_done=${shardsDone}/${args.shardCount}`);
  }, 30_000);

  let byShard: Summary['byShard'];
  try {
    byShard = await Promise.all(shardPromises);
  } finally {
    clearInterval(progress);
  }

  await new Promise<void>((resolveDone, rejectDone) => {
    stream.end((err?: Error | null) => {
      if (err) rejectDone(err);
      else resolveDone();
    });
  });

  const rawResults = await import('node:fs/promises')
    .then((fs) => fs.readFile(rawJsonl, 'utf-8'))
    .then((contents) => contents.trim() ? contents.trim().split('\n').map((line) => JSON.parse(line) as SandboxResult) : []);

  const successes = rawResults.filter((result) => result.status === 'success');
  const statusCounts: Record<Status, number> = {
    success: 0,
    failed: 0,
    readiness_failed: 0,
    destroy_failed: 0,
  };

  for (const result of rawResults) {
    statusCounts[result.status]++;
  }

  const summary: Summary = {
    runId,
    provider: 'tensorlake',
    shardCount: args.shardCount,
    sandboxesPerShard: args.sandboxesPerShard,
    totalSandboxes,
    sandboxOptions: configuredSandboxOptions,
    startedAt,
    completedAt: new Date().toISOString(),
    wallClockMs: Math.round(performance.now() - wallStart),
    statusCounts,
    createMs: distribution(successes.map((result) => result.createMs)),
    commandMs: distribution(successes.flatMap((result) => result.commandMs === null ? [] : [result.commandMs])),
    ttiMs: distribution(successes.flatMap((result) => result.ttiMs === null ? [] : [result.ttiMs])),
    byShard,
    files: {
      rawJsonl,
      summaryJson,
    },
  };

  writeFileSync(summaryJson, `${JSON.stringify(summary, null, 2)}\n`);
  console.log(`complete success=${statusCounts.success}/${totalSandboxes} wall_clock_ms=${summary.wallClockMs}`);
  console.log(`summary=${summaryJson}`);
  console.log(`raw=${rawJsonl}`);
}

main().catch((err) => {
  console.error(err instanceof Error ? err.stack || err.message : err);
  process.exit(1);
});
