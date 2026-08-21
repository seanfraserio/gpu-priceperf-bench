# gppb-sink

Write-authenticated result sink. Rented GPU instances `PUT` their `BenchResult`
JSON here; the Worker checks a bearer token and stores the object in R2.

The instances power themselves off the moment a run finishes, so this is the
only place a result survives — an unreachable sink means a paid run produces
nothing.

## Deploy

    npx wrangler r2 bucket create gppb-results
    npx wrangler secret put SINK_TOKEN      # same value goes to the instance
    npx wrangler deploy

## Use

The runner reads both from its environment:

    SINK_URL=https://gppb-sink.<subdomain>.workers.dev
    SINK_TOKEN=<the secret>

`run_id` becomes the object key (`<run_id>.json`), constrained to
`[A-Za-z0-9][A-Za-z0-9._-]*\.json` — no slashes, no traversal. Repeat PUTs of
the same `run_id` overwrite by design: each carries a more complete result than
the last, so a preempted run keeps whatever it managed to publish.

## Pull results down

Reads are authenticated too. List what a run uploaded, then fetch it:

    curl -H "Authorization: Bearer $SINK_TOKEN" https://gppb-sink.<sub>.workers.dev/_list
    curl -H "Authorization: Bearer $SINK_TOKEN" \
      https://gppb-sink.<sub>.workers.dev/<run_id>.json -o results/<run_id>.json
