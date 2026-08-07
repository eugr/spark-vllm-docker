# Visualizing vLLM metrics with Prometheus and Grafana

vLLM exposes Prometheus metrics from the OpenAI-compatible server at
`/metrics`. This setup runs Prometheus and Grafana alongside an existing
vLLM container, providing a web UI for request latency, throughput, KV-cache
usage, and speculative-decoding performance.

This example assumes vLLM is listening on port `8000` on the Docker host.
Confirm the endpoint first:

```bash
curl http://127.0.0.1:8000/metrics
```

## Start Prometheus and Grafana

Create `prometheus.yaml`:

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: vllm
    static_configs:
      - targets:
          - host.docker.internal:8000
```

Create `docker-compose.metrics.yaml` in the same directory:

```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    restart: unless-stopped
    extra_hosts:
      - host.docker.internal:host-gateway
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yaml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus

  grafana:
    image: grafana/grafana:latest
    restart: unless-stopped
    depends_on:
      - prometheus
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  prometheus-data:
  grafana-data:
```

Start both services:

```bash
docker compose -f docker-compose.metrics.yaml up -d
```

Open the services in a browser, replacing `<spark-ip>` with the LAN address
of the head node:

- Grafana: `http://<spark-ip>:3000`
- Prometheus: `http://<spark-ip>:9090`

The initial Grafana login is `admin` / `admin`. Grafana will ask you to
change the password after the first login.

## Connect Grafana to Prometheus

1. In Grafana, open **Connections > Data sources** and add **Prometheus**.
2. Set the Prometheus server URL to `http://prometheus:9090`.
3. Select **Save & test**.
4. Check `http://<spark-ip>:9090/targets`; the `vllm` target should be `UP`.

The upstream vLLM repository provides an example dashboard that can be
imported into Grafana:

<https://docs.vllm.ai/en/latest/examples/observability/prometheus_grafana/>

## Speculative decoding and MTP panels

The general vLLM dashboard may not include speculative-decoding panels. Add
Grafana panels with the following PromQL queries.

### Draft-token acceptance rate

```promql
100 *
sum(rate(vllm:spec_decode_num_accepted_tokens_total[5m]))
/
sum(rate(vllm:spec_decode_num_draft_tokens_total[5m]))
```

Use a Gauge or Time series visualization with the unit set to percent.

### Mean acceptance length

This convention includes the target/bonus token emitted by a verification
step:

```promql
1 +
sum(rate(vllm:spec_decode_num_accepted_tokens_total[5m]))
/
sum(rate(vllm:spec_decode_num_drafts_total[5m]))
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
100 *
sum by (position) (
  rate(vllm:spec_decode_num_accepted_tokens_per_pos_total[5m])
)
/
scalar(
  sum(rate(vllm:spec_decode_num_drafts_total[5m]))
)
```

Use a Bar chart or Time series visualization and set the legend to
`Position {{position}}`.

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
