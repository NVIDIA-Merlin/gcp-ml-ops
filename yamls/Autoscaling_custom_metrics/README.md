# Custom metrics API server

The manifests require cert-manager with its CA injector enabled. The certificate
resources create a dedicated CA and a renewable serving certificate for
`api.custom-metrics.svc`; the CA injector writes the trust bundle into the
registered API service.

The Prometheus endpoint must serve HTTPS with a certificate issued by a trusted
CA. The deployment defaults to
`https://prometheus-operated.monitoring.svc.cluster.local:9090`. Update the
`--prometheus-url` argument if the service has a different DNS name, then make
the Prometheus CA available to the adapter:

```sh
kubectl create namespace custom-metrics --dry-run=client -o yaml | kubectl apply -f -
kubectl -n custom-metrics create configmap prometheus-client-ca \
  --from-file=ca.crt=/path/to/prometheus-ca.crt
```

Apply the resources after cert-manager and the Prometheus CA are ready:

```sh
kubectl apply -f 0_custom-metrics-certificates.yaml
kubectl apply -f 1_custom-metric-server-config.yaml
kubectl apply -f 3_custom-metrics-server-rbac.yaml
kubectl apply -f 2_custom-metric-server.yaml
kubectl apply -f 4_triton-hpa.yaml
```

The adapter:

- runs as a non-root user with a read-only root filesystem and no Linux
  capabilities;
- serves the aggregated API over TLS 1.2 or newer using the renewable
  certificate;
- verifies the Prometheus server certificate using `prometheus-client-ca`;
- exposes port 6443 inside the pod so it does not need privilege to bind a low
