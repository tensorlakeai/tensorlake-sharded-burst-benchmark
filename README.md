# Tensorlake Sharded Burst Benchmark

This repo has two Tensorlake benchmark modes. The intended distributed mode is `bench:tensorlake:nested`: it creates Tensorlake shard sandboxes first, then each shard sandbox creates `100` child sandboxes in parallel and runs `node -v` in each child.

Default scale:

- `1000` shards
- `100` sandboxes per shard
- `100000` total sandbox creations

Each result records:

- `createMs`: time to create the sandbox
- `commandMs`: time to run `node -v` after creation
- `ttiMs`: `createMs + commandMs`

## Setup

```sh
cp env.example .env
npm install
```

Set `TENSORLAKE_API_KEY` in `.env`.

## Run

```sh
npm run bench:tensorlake:nested
```

For a smaller smoke test:

```sh
npm run bench:tensorlake:nested -- --shards 2 --children-per-shard 3
```

Outputs are written to `results/<run-id>/raw.jsonl`, `results/<run-id>/shards.jsonl`, and `results/<run-id>/summary.json`.

## Options

```sh
npm run bench:tensorlake:nested -- \
  --shards 1000 \
  --children-per-shard 100 \
  --shard-create-concurrency 100 \
  --shard-command-timeout-ms 300000
```

The older `bench:tensorlake` command is a flat local burst harness. It is useful for small smoke tests, but it is not the correct topology for the 1000-shard benchmark because it makes the laptop initiate every child sandbox request.

The nested worker uses Tensorlake APIs directly from inside each shard sandbox, so the laptop only creates the shard sandboxes and starts one worker command per shard.
