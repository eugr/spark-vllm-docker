# Visualizing vLLM metrics with Prometheus and Grafana

vLLM exposes Prometheus metrics from the OpenAI-compatible server at
`/metrics`. This setup runs Prometheus and Grafana alongside an existing
vLLM container, providing a web UI for request latency, throughput, KV-cache
usage, and speculative-decoding performance.

The repository includes a ready-to-run monitoring stack:

- `docker-compose.metrics.yaml` runs Prometheus and Grafana with persistent
  named volumes.
- `prometheus.yaml` scrapes the vLLM server from the Docker host every five
  seconds.
- `prometheus-rules.yaml` records one-minute averages for prompt, generation,
  and total token throughput, plus speculative-decoding metrics (acceptance
  rate, mean accepted length, draft and accepted token rates, and
  per-position acceptance).
- `grafana/provisioning/` configures Prometheus as Grafana's default data source
  and loads dashboards from disk.
- `grafana/dashboards/vllm-throughput.json` defines the default **vLLM
  Throughput** dashboard.
- `grafana/dashboards/vllm-spec-decode.json` defines the **vLLM Speculative
  Decoding** dashboard.

The default configuration assumes vLLM is listening on port `8000` on the
Docker host. Confirm the endpoint first:

```bash
curl http://127.0.0.1:8000/metrics
```

## Start Prometheus and Grafana

From the repository root, start both services:

```bash
docker compose -f docker-compose.metrics.yaml up -d
```

Run the same command after changing the Compose file or provisioning files.
Compose recreates affected containers when their configuration changes; a plain
`docker compose restart` does not apply new mounts or environment settings.

If vLLM uses a different host port, update the target in `prometheus.yaml`
before starting the stack. You can inspect service state and logs with:

```bash
docker compose -f docker-compose.metrics.yaml ps
docker compose -f docker-compose.metrics.yaml logs prometheus grafana
```

Open the services in a browser, replacing `<spark-ip>` with the LAN address
of the head node:

- Grafana: `http://<spark-ip>:3000`
- Prometheus: `http://<spark-ip>:9090`

The initial Grafana login is `admin` / `admin`. Grafana will ask you to
change the password after the first login. The provisioned **vLLM Throughput**
dashboard opens as the home dashboard and shows one-minute averages for prompt,
generation, and total token throughput.

## Provisioned data source and dashboard

The Compose stack provisions Prometheus as Grafana's default data source at
startup. No manual data-source setup is required. Check
`http://<spark-ip>:9090/targets`; the `vllm` target should be `UP`.

The dashboard uses these recording rules from `prometheus-rules.yaml`:

| Recorded metric | Meaning |
| --- | --- |
| `vllm:generation_tokens_per_second:rate1m` | Output tokens per second |
| `vllm:prompt_tokens_per_second:rate1m` | Input/prompt tokens per second |
| `vllm:total_tokens_per_second:rate1m` | Combined input and output tokens per second |

Inspect loaded rules at `http://<spark-ip>:9090/rules`, or enter one of the
recorded metric names in the Prometheus query UI. Prometheus needs at least two
samples before a rate appears, and the panels remain at zero while vLLM is
idle. The dashboard refreshes every five seconds and displays the most recent
one-minute averages plus a 15-minute history.

The provisioned dashboard can be edited in Grafana. For a durable,
version-controlled change, export the updated dashboard JSON and replace
`grafana/dashboards/vllm-throughput.json`; later provisioning-file changes can
overwrite a dashboard saved only in Grafana's database.

The upstream vLLM repository provides an example dashboard that can be
imported into Grafana:

<https://docs.vllm.ai/en/latest/examples/observability/prometheus_grafana/>

## Speculative decoding and MTP panels

The provisioned **vLLM Speculative Decoding** dashboard
(`grafana/dashboards/vllm-spec-decode.json`) covers these metrics out of the
box. It appears in the **vLLM** folder and queries the `vllm:spec_decode_*`
recording rules from the `vllm-spec-decode` group in `prometheus-rules.yaml`:

- **Draft acceptance rate** — fraction of draft tokens accepted
  (`vllm:spec_decode_acceptance_rate:rate1m`)
- **Mean accepted length** — tokens emitted per verification step, including
  the bonus token (`vllm:spec_decode_mean_accepted_length:rate1m`); with N
  speculative tokens it ranges from 1.0 to N+1
- **Draft tokens / second** (`vllm:spec_decode_draft_tokens_per_second:rate1m`)
- **Draft vs accepted tokens / second** — the gap between the two series is
  wasted draft compute
- **Acceptance rate by draft position** — per-position quality of the MTP
  cascade (`vllm:spec_decode_acceptance_rate_by_pos:rate1m`)

The stat panels query the recorded metrics as instant values. The ratio
rules (acceptance rate, mean accepted length, per-position) evaluate to no
value once traffic stops, so those panels show **No value** shortly after
the last request, when the last recorded sample leaves Prometheus'
five-minute lookback window; this avoids a misleading 100% acceptance rate
at idle. The token-rate rules have no such guard and keep recording zero
while vLLM is running, so the draft and accepted token panels show 0 at
idle.

If you add or replace dashboard files on disk, restart the Grafana container
for the file provider to reload them:

```bash
docker compose -f docker-compose.metrics.yaml restart grafana
```

If you prefer to build your own panels directly on the raw counters (for
example with a longer rate window), the following PromQL queries are
equivalent to the recording rules:

### Draft-token acceptance rate

```promql
(
  100 *
  sum(rate(vllm:spec_decode_num_accepted_tokens_total[5m]))
  /
  sum(rate(vllm:spec_decode_num_draft_tokens_total[5m]))
)
and on()
(
  sum(rate(vllm:spec_decode_num_draft_tokens_total[5m])) > 0
)
```

Use a Gauge or Time series visualization with the unit set to percent.

### Mean acceptance length

This convention includes the target/bonus token emitted by a verification
step:

```promql
(
  1 +
  sum(rate(vllm:spec_decode_num_accepted_tokens_total[5m]))
  /
  sum(rate(vllm:spec_decode_num_drafts_total[5m]))
)
and on()
(
  sum(rate(vllm:spec_decode_num_drafts_total[5m])) > 0
)
```

### Draft and accepted tokens per second

Draft tokens proposed:

```promql
sum(rate(vllm:spec_decode_num_draft_tokens_total[5m]))
```

Draft tokens accepted:

```promql
sum(rate(vllm:spec_decode_num_accepted_tokens_total[5m]))
```

### Acceptance rate by draft position

```promql
(
  100 *
  sum by (position) (
    rate(vllm:spec_decode_num_accepted_tokens_per_pos_total[5m])
  )
  /
  scalar(
    sum(rate(vllm:spec_decode_num_drafts_total[5m]))
  )
)
and on()
(
  sum(rate(vllm:spec_decode_num_drafts_total[5m]))
  > 0
)
```

The provisioned dashboard renders this metric as a Bar gauge with one bar
per draft position. If you build your own panel, use a Bar gauge or Time
series visualization and set the legend to `Position {{position}}`.

## Generate test traffic

Prometheus counters change only while requests are being processed. Generate
a sufficiently long response while viewing the dashboard, for example:

```bash
uvx llama-benchy \
  --base-url http://127.0.0.1:8000/v1 \
  --model google/gemma-4-26B-A4B-it \
  --pp 2048 \
  --tg 512
```

If a new panel initially shows no data, wait for at least two Prometheus
scrapes. You can temporarily change `[5m]` to `[30s]` while testing.

Acceptance rate alone is not the speculative-decoding speedup. Compare the
same llama-benchy or `vllm bench serve` workload with speculative decoding
enabled and disabled to measure the actual change in tokens per second and
latency.

## Stop the monitoring services

```bash
docker compose -f docker-compose.metrics.yaml down
```

The named volumes preserve Prometheus history and Grafana configuration. Add
`-v` only when you intentionally want to delete that stored monitoring data.
